#include <bits/stdc++.h>
#include <sys/ioctl.h>

//lib
#include <rgb.h>
#include <img.h>
#include <termios.h>

//glm
#define GLM_FORCE_AVX2
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

struct termios g_original_termios;

struct {
    int x = 0;
    int y = 0;
} mousePos;

constexpr float FP_EPSILON = 1e-6f;
constexpr int CORES = 12;
const int threshold = 0;

const int TARGET_FPS = 3000; //-1 for uncapped fps
const chrono::duration<double, std::nano> FRAME_DURATION(1000000000.0 / TARGET_FPS);

vector<vector<color>>FRAME_BUFFER;

int SCREEN_WIDTH,SCREEN_HEIGHT;
chrono::steady_clock::time_point start_time;
chrono::steady_clock::time_point delta_time_clock;

array<binary_semaphore,CORES>semaphore_full=[]<size_t...Is>(index_sequence<Is...>){return array<binary_semaphore,sizeof...(Is)>{((void)Is,binary_semaphore{0})...};}(make_index_sequence<CORES>());
array<binary_semaphore,CORES>semaphore_empty=[]<size_t...Is>(index_sequence<Is...>){return array<binary_semaphore,sizeof...(Is)>{((void)Is,binary_semaphore{0})...};}(make_index_sequence<CORES>());

void InitTerminalInput() {
    tcgetattr(STDIN_FILENO, &g_original_termios);
    struct termios raw_termios = g_original_termios;
    raw_termios.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw_termios.c_cc[VMIN] = 0;
    raw_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw_termios);
    cout << "\x1b[?1003h" << flush; 
}

void RestoreTerminal() {
    std::cout << "\x1b" "c" << std::flush;
}

bool ReadInput() {
    constexpr int READ_BUFFER_SIZE = 128;
    char buf[READ_BUFFER_SIZE];
    
    // Read all available bytes (non-blocking)
    ssize_t nread = read(STDIN_FILENO, buf, sizeof(buf));
    
    if (nread <= 0) {
        return false; // No input
    }
    
    for (ssize_t i = 0; i < nread; ++i) {
        if (buf[i] == 'c'||buf[i] == 'q') {
            return true;
        }
        if (i + 2 < nread && buf[i] == '\x1b' && buf[i+1] == '[' && buf[i+2] == 'M') {
            if (i + 5 < nread) {
                mousePos.x = (unsigned char)buf[i+4] - 33;
                mousePos.y = SCREEN_HEIGHT - (unsigned char)buf[i+5] + 32;
                i += 5; 
            }
        }
    }
    return false;
}

template <typename T>
class Matrix {
    int size_x, size_y;
    std::vector<T> data_;
public:
    Matrix(int cols_, int rows_)
        : size_x(cols_), size_y(rows_), data_(rows_ * cols_) {}

    inline T& operator()(int r, int c) noexcept {
        return data_[r * size_x + c];
    }
};

void getTerminalSize(int &x, int &y) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    y = w.ws_row;
    x = w.ws_col;
}


///////////////////////////////
// ↓ RENDERING STUFF ↓
inline float PointIsOnRightSideOfLine(glm::vec2 a, glm::vec2 b, glm::vec2 p){ // res<0 == true
    b-=a;
    p-=a;
    return (b.x*p.y) - (p.x*b.y);
}

inline bool EdgeIsTopLeft(const glm::vec2 &a, const glm::vec2 &b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    return (dy > 0.0f) || (fabsf(dy) < FP_EPSILON && dx < 0.0f);
}

inline bool PointIsInsideTriangle(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 p){
    float ab = PointIsOnRightSideOfLine(a, b, p);
    float bc = PointIsOnRightSideOfLine(b, c, p);
    float ca = PointIsOnRightSideOfLine(c, a, p);

    return (ab<0||(EdgeIsTopLeft(a,b)&&ab<FP_EPSILON)) &&
           (bc<0||(EdgeIsTopLeft(b,c)&&bc<FP_EPSILON)) &&
           (ca<0||(EdgeIsTopLeft(c,a)&&ca<FP_EPSILON));
}

inline float AreaDouble(glm::vec2 a, glm::vec2 b, glm::vec2 c){
    return fabs(a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y));
}

void build_line (int yb, int ye, vector<string>& buffer, int id) {
    for(;;){
        semaphore_empty[id].acquire();

        const int sw = SCREEN_WIDTH;// sh = SCREEN_HEIGHT;
        color last_pixel(0,0,0), last_pixel2(0,0,0);
        

        for(int screen_y = yb; screen_y < ye; screen_y++){
            string &line = buffer[buffer.size() - 1 - screen_y];
            line.clear();

            char numbuf[16];
            for(int screen_x = 0; screen_x < sw; screen_x++){
                color pixel = FRAME_BUFFER[screen_x][screen_y*2].Clamp();
                color pixel2 = FRAME_BUFFER[screen_x][screen_y*2+1].Clamp();

                if(screen_x==mousePos.x&&screen_y==mousePos.y) pixel = color(1,0,0);

                if(screen_x == 0){
                    last_pixel = pixel;
                    last_pixel2 = pixel2;
                    line.append("\x1b[38;2;");
                    auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.r*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.push_back(';');
                    res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.g*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.push_back(';');
                    res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.b*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.append("m");
                    last_pixel = pixel;
                    line.append("\x1b[48;2;");
                    res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.r*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.push_back(';');
                    res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.g*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.push_back(';');
                    res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.b*255)));
                    line.append(numbuf, res.ptr - numbuf);
                    line.append("m");
                } else {
                    int dist_sq  = ColorDifferenceSquared(pixel, last_pixel);
                    int dist_sq2 = ColorDifferenceSquared(pixel2, last_pixel2);

                    if(dist_sq > threshold){
                        line.append("\x1b[38;2;");
                        auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.r*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.push_back(';');
                        res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.g*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.push_back(';');
                        res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel.b*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.append("m");
                        last_pixel = pixel;
                    }
                    if(dist_sq2 > threshold){
                        line.append("\x1b[48;2;");
                        auto res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.r*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.push_back(';');
                        res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.g*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.push_back(';');
                        res = std::to_chars(numbuf, numbuf + sizeof(numbuf), (int)(max(1.0f,pixel2.b*255)));
                        line.append(numbuf, res.ptr - numbuf);
                        line.append("m");
                        last_pixel2 = pixel2;
                    }
                }

                line+="▄";
            } // x
        } // y

        semaphore_full[id].release();
    }
}
// ↑ RENDERING STUFF ↑
///////////////////////////////



Matrix<glm::vec2> vel(SCREEN_WIDTH+2, SCREEN_HEIGHT*2+2);
Matrix<glm::vec2> prev_vel(SCREEN_WIDTH+2, SCREEN_HEIGHT*2+2);
Matrix<float> dens(SCREEN_WIDTH+2, SCREEN_HEIGHT*2+2);
Matrix<float> prev_dens(SCREEN_WIDTH+2, SCREEN_HEIGHT*2+2);

//       FLUID SIM FUNCTIONS
//////////////////////////////////

void AddSource(Matrix<float> &densMat, Matrix<float> &srcMat, float dTime){

}




//////////////////////////////////



int main(){
    ios::sync_with_stdio(false);
    cout << "\x1b[?25l"; //hide cursor

    //INPUT setup
    /////////////////////////////

    InitTerminalInput();

    /////////////////////////////

    getTerminalSize(SCREEN_WIDTH, SCREEN_HEIGHT);

    FRAME_BUFFER = vector<vector<color>>(SCREEN_WIDTH,vector<color>(SCREEN_HEIGHT*2,{0,0,0}));
    
    string output;
    int cur_fps=0;
    chrono::steady_clock::time_point fps_timer = std::chrono::steady_clock::now();
    start_time = std::chrono::steady_clock::now();

    int renderheight = SCREEN_HEIGHT-1;
    int block_size = renderheight / CORES;

    vector<string> buffer(SCREEN_HEIGHT);
    for(auto &line:buffer) line.reserve(SCREEN_WIDTH * 34 + 32);
    array<thread,CORES> threads;

    output.reserve(SCREEN_WIDTH * (SCREEN_HEIGHT - 1) * 34 + 196);

    for(int y = 0; y < CORES; y++){
        threads[y] = thread(build_line, y*block_size, (y==CORES-1?renderheight:(y+1)*block_size), std::ref(buffer), y);
    }

    //         AREA CAZZEGGIO
    ////////////////////////////////

    

    ////////////////////////////////

    chrono::time_point<chrono::steady_clock> frame_start, frame_end;
    chrono::duration<double, std::milli> frame_dur;

    for(int cur_frame = 0;;cur_frame++){

        frame_start = chrono::steady_clock::now();

        delta_time_clock = std::chrono::steady_clock::now();
        int delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(delta_time_clock - fps_timer).count();
        if (delta_time >= 250) {
            cur_fps = cur_frame / (delta_time / 1000.0f);
            cur_frame = 0;
            fps_timer = delta_time_clock;
        }

        //            INPUT
        ////////////////////////////////////////////

        if(ReadInput()){
            RestoreTerminal();
            return 0;
        }

        //            UPDATE LOOP
        ////////////////////////////////////////////

        std::chrono::duration<float> curTime = delta_time_clock - start_time;
        

        ////////////////////////////////////////////
        //           RENDERING

        for (auto &col : FRAME_BUFFER)
            std::fill(col.begin(), col.end(), color());

        for(auto &s:semaphore_empty) s.release();
    

        output.clear();

        output += "\x1b[H\x1b[?25l";
        output += "\x1b[39;49m" + to_string(SCREEN_WIDTH) + "x" + to_string(SCREEN_HEIGHT*2) + " fps:" + to_string(cur_fps) + " mouse: [" + to_string(mousePos.x) + ":" + to_string(mousePos.y) + "]\x1b[K\n";

        for(auto &s:semaphore_full) s.acquire();

        for(auto &i:buffer) output += i;
        cout << output;
        cout.flush();

        frame_end = chrono::steady_clock::now();
        frame_dur = frame_end - frame_start;
        auto target_frame_end = frame_start + FRAME_DURATION;

        if (TARGET_FPS!=-1 && frame_end < target_frame_end) {
            auto time_to_wait = target_frame_end - frame_end;
            auto yield_threshold = chrono::milliseconds(1);

            if (time_to_wait > yield_threshold) {
                this_thread::sleep_for(time_to_wait - yield_threshold);
            }

            while (chrono::steady_clock::now() < target_frame_end) {
                this_thread::yield();
            }
        }
    }

    return 0;
}