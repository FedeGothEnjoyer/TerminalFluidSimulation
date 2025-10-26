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

const int TARGET_FPS = 60; //-1 for uncapped fps
const chrono::duration<double, std::nano> FRAME_DURATION(1000000000.0 / TARGET_FPS);
const float FIXED_DELTA_TIME = 1.0f/TARGET_FPS;



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

char ReadInput() {
    constexpr int READ_BUFFER_SIZE = 128;
    char buf[READ_BUFFER_SIZE];
    
    // Read all available bytes (non-blocking)
    ssize_t nread = read(STDIN_FILENO, buf, sizeof(buf));
    
    if (nread <= 0) {
        return false; // No input
    }
    
    for (ssize_t i = 0; i < nread; ++i) {
        if (buf[i] == 'c'||buf[i] == 'q') {
            return buf[i];
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
public:
    int size_x, size_y;
    std::vector<T> data_;
    Matrix(int cols_, int rows_)
        : size_x(cols_), size_y(rows_), data_(rows_ * cols_) {}

    inline T& operator()(int r, int c) noexcept {
        return data_[c * size_x + r];
    }
};

int simSizeX;
int simSizeY;

Matrix<float> dens(0,0);

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

                auto d_val = min(10.0f,dens(screen_x+1,screen_y*2+1))/10.0f;
                pixel = {d_val,d_val,d_val};

                d_val = min(10.0f,dens(screen_x+1,screen_y*2+2))/10.0f;
                pixel2 = {d_val,d_val,d_val};


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





//       FLUID SIM FUNCTIONS
//////////////////////////////////

void AddSource(Matrix<float> &densMat, Matrix<float> &srcMat, float dTime){
    for(int i = 0; i < densMat.size_x*densMat.size_y; i++) densMat.data_[i] += srcMat.data_[i]*dTime;
}

void SetBound(int b, Matrix<float> &x){
    for (int i=1 ; i<=simSizeY ; i++ ) {
        x(0 ,i) = b==1 ? -x(1,i) : x(1,i);
        x(simSizeX+1,i) = b==1 ? -x(simSizeX,i) : x(simSizeX,i);
    }
    for(int i=1 ; i<=simSizeX ; i++ ){
        x(i,0 ) = b==2 ? -x(i,1) : x(i,1);
        x(i,simSizeY+1) = b==2 ? -x(i,simSizeY) : x(i,simSizeY);
    }
    x(0 ,0 ) = 0.5*(x(1,0 )+x(0 ,1));
    x(0 ,simSizeY+1) = 0.5*(x(1,simSizeY+1)+x(0 ,simSizeY ));
    x(simSizeX+1,0 ) = 0.5*(x(simSizeX,0 )+x(simSizeX+1,1));
    x(simSizeX+1,simSizeY+1) = 0.5*(x(simSizeX,simSizeY+1)+x(simSizeX+1,simSizeY ));
}

void Diffuse(int b, Matrix<float> &densMat, Matrix<float> &prevDensMat, float diff, float dTime){
    float a = dTime*diff*(densMat.size_x-2)*(densMat.size_y-2);
    for(int k = 0; k < 20; k++){
        for(int i = 1; i <= (densMat.size_x-2); i++){
            for(int j = 1; j <= (densMat.size_y-2); j++){
                densMat(i,j) = (prevDensMat(i,j) + a*(densMat(i-1,j)+densMat(i+1,j)+densMat(i,j-1)+densMat(i,j+1)))/(1+4*a);
            }
        }
        SetBound(b, densMat);
    }
}

void Advect(int b, Matrix<float> &densMat, Matrix<float> &prevDensMat, Matrix<float> &velMatX, Matrix<float> &velMatY, float dTime){
    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1;
    float dt0x = dTime*simSizeX;
    float dt0y = dTime*simSizeY;
    for(int i = 1; i <= simSizeX; i++){
        for(int j = 1; j <= simSizeY; j++){
            x = i-dt0x*velMatX(i,j); y = j-dt0y*velMatY(i,j);
            if (x<0.5) x=0.5; if (x>simSizeX+0.5) x=simSizeX + 0.5; i0=(int)x; i1=i0+ 1;
            if (y<0.5) y=0.5; if (y>simSizeY+0.5) y=simSizeY + 0.5; j0=(int)y; j1=j0+1;
            s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
            densMat(i,j) = s0*(t0*prevDensMat(i0,j0)+t1*prevDensMat(i0,j1))+s1*(t0*prevDensMat(i1,j0)+t1*prevDensMat(i1,j1));
        }
    }
    SetBound(b, densMat);
}

void Project(Matrix<float> &velMatX,Matrix<float> &velMatY, Matrix<float> &v_p,Matrix<float> &v_div){
    float hx = 1.0/simSizeX;
    float hy = 1.0/simSizeY;
    
    for(int i = 1; i <= simSizeX; i++){
        for(int j = 1; j <= simSizeY; j++){
            v_div(i,j) = -0.5*hx*(velMatX(i+1,j)-velMatX(i-1,j)+velMatY(i,j+1)-velMatY(i,j-1));
            v_p(i,j) = 0;
        }
    }
    SetBound(0, v_div); SetBound(0, v_p);
    for(int k=0; k < 20; k++){
        for (int i=1 ; i<=simSizeX ; i++ ) {
            for (int j=1 ; j<=simSizeY ; j++ ) {
                v_p(i,j) = (v_div(i,j)+v_p(i-1,j)+v_p(i+1,j)+v_p(i,j-1)+v_p(i,j+1))/4;
            }
        }
        SetBound(0, v_p);
    }
    for (int i=1 ; i<=simSizeX ; i++ ) {
        for (int j=1 ; j<=simSizeY ; j++ ) {
            velMatX(i,j) -= 0.5*(v_p(i+1,j)-v_p(i-1,j))/hx;
            velMatY(i,j) -= 0.5*(v_p(i,j+1)-v_p(i,j-1))/hy;
        }
    }
    SetBound(1, velMatX); SetBound(2, velMatY);
}

void DensityStep(Matrix<float> &densMat, Matrix<float> &startDensMat, Matrix<float> &velMatX, Matrix<float> &velMatY, float diff, float dTime){
    AddSource(densMat, startDensMat, dTime);
    swap(startDensMat, densMat); Diffuse(0, densMat, startDensMat, diff, dTime);
    swap(startDensMat, densMat); Advect(0, densMat, startDensMat, velMatX, velMatY, dTime);
}

void VelocityStep(Matrix<float> &velMatX, Matrix<float> &velMatY, Matrix<float> &startVelMatX, Matrix<float> &startVelMatY, float visc, float dTime){
    AddSource(velMatX, startVelMatX, dTime); AddSource(velMatY, startVelMatY, dTime);
    swap(startVelMatX, velMatX); Diffuse(1, velMatX, startVelMatX, visc, dTime);
    swap(startVelMatY, velMatY); Diffuse(2, velMatY, startVelMatY, visc, dTime);
    Project(velMatX, velMatY, startVelMatX, startVelMatY);
    swap(startVelMatX, velMatX); swap(startVelMatY, velMatY);
    Advect(1, velMatX, startVelMatX, startVelMatX, startVelMatY, dTime);
    Advect(2, velMatY, startVelMatY, startVelMatX, startVelMatY, dTime);
    Project(velMatX, velMatY, startVelMatX, startVelMatY);
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
    simSizeX = SCREEN_WIDTH;
    simSizeY = SCREEN_HEIGHT*2;

    Matrix<float> vel_x(simSizeX+2, simSizeY+2);
    Matrix<float> vel_y(simSizeX+2, simSizeY+2);
    Matrix<float> prev_vel_x(simSizeX+2, simSizeY+2);
    Matrix<float> prev_vel_y(simSizeX+2, simSizeY+2);
    dens = Matrix<float>(simSizeX+2, simSizeY+2);
    Matrix<float> prev_dens(simSizeX+2, simSizeY+2);
    

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

        char ch = ReadInput();
        if(ch=='q'){
            RestoreTerminal();
            return 0;
        }

        //            UPDATE LOOP
        ////////////////////////////////////////////

        std::chrono::duration<float> curTime = delta_time_clock - start_time;
        
        //swap(prev_dens,dens);
        fill(prev_dens.data_.begin(), prev_dens.data_.end(), 0.0f);
        fill(prev_vel_x.data_.begin(), prev_vel_x.data_.end(), 0.0f);
        fill(prev_vel_y.data_.begin(), prev_vel_y.data_.end(), 0.0f);

        //demo wind tunnel
        prev_vel_y(105,30) = 2000.0f;
        prev_vel_y(40,70) = -2000.0f;
        prev_vel_y(165,70) = -2000.0f;
        prev_vel_x(80, 80) = -1000.0f;
        prev_vel_x(120, 80) = 1000.0f;
        prev_vel_x(80, 20) = 1000.0f;
        prev_vel_x(120, 20) = -1000.0f;

        prev_dens(mousePos.x+1,mousePos.y*2+1)=5000.0f;
        
        //prev_vel_y(60,10) = 100.0f;

        VelocityStep(vel_x,vel_y,prev_vel_x,prev_vel_y,0.001f,FIXED_DELTA_TIME);
        DensityStep(dens, prev_dens, vel_x, vel_y, 0.0001f, FIXED_DELTA_TIME);


        ////////////////////////////////////////////
        //           RENDERING

        for (auto &col : FRAME_BUFFER)
            std::fill(col.begin(), col.end(), color());

        for(auto &s:semaphore_empty) s.release();
    

        output.clear();

        output += "\x1b[H\x1b[?25l";
        output += "\x1b[39;49m" + to_string(SCREEN_WIDTH) + "x" + to_string(SCREEN_HEIGHT*2) + " fps:" + to_string(cur_fps) + " mouse: [" + to_string(mousePos.x) + ":" + to_string(mousePos.y) + "] (" + to_string(dens(mousePos.x,mousePos.y*2)) + ")\x1b[K\n";

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