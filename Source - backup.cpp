#include <stdio.h>
#include <math.h>
#include "Eigen/Dense"
#include "SDL.h"
#include "glad/glad.h"
#include <GL/gl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "Shader.h"
#include "stb_image.h"
#include <fstream>
#define ArrayCount(X) (sizeof(X)/sizeof(*X))

#ifndef GL_STACK_OVERFLOW
#define GL_STACK_OVERFLOW 0x0503
#endif
#ifndef GL_STACK_UNDERFLOW
#define GL_STACK_UNDERFLOW 0x0504
#endif
#ifndef GL_OUT_OF_MEMORY
#define GL_OUT_OF_MEMORY 0x0505
#endif
#ifndef GL_INVALID_FRAMEBUFFER_OPERATION
#define GL_INVALID_FRAMEBUFFER_OPERATION 0x0506
#endif
#ifndef GL_CONTEXT_LOST
#define GL_CONTEXT_LOST 0x0507
#endif
#ifndef GL_TABLE_TOO_LARGE1
#define GL_TABLE_TOO_LARGE1 0x8031
#endif

static const char* ErrorStringOpenGL(GLenum Error) {
    switch (Error) {
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST";
    case GL_TABLE_TOO_LARGE1: return "GL_TABLE_TOO_LARGE1";
    default: return "unknwon";
    }
}

static const char* ErrorDescriptionOpenGL(GLenum Error) {
    // NOTE: Taken from https://www.khronos.org/opengl/wiki/OpenGL_Error 
    switch (Error) {
    case GL_INVALID_ENUM: return "Given when an enumeration parameter is not a legal enumeration for that function. This is given only for local problems; if the spec allows the enumeration in certain circumstances, where other parameters or state dictate those circumstances, then GL_INVALID_OPERATION is the result instead.";
    case GL_INVALID_VALUE: return "Given when a value parameter is not a legal value for that function. This is only given for local problems; if the spec allows the value in certain circumstances, where other parameters or state dictate those circumstances, then GL_INVALID_OPERATION is the result instead.";
    case GL_INVALID_OPERATION: return "Given when the set of state for a command is not legal for the parameters given to that command. It is also given for commands where combinations of parameters define what the legal parameters are.";
    case GL_STACK_OVERFLOW: return "Given when a stack pushing operation cannot be done because it would overflow the limit of that stack's size.";
    case GL_STACK_UNDERFLOW: return "Given when a stack popping operation cannot be done because the stack is already at its lowest point.";
    case GL_OUT_OF_MEMORY: return "Given when performing an operation that can allocate memory, and the memory cannot be allocated. The results of OpenGL functions that return this error are undefined; it is allowable for partial execution of an operation to happen in this circumstance.";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "Given when doing anything that would attempt to read from or write/render to a framebuffer that is not complete.";
    case GL_CONTEXT_LOST: return "Given if the OpenGL context has been lost, due to a graphics card reset (with OpenGL 4.5 or ARB_KHR_robustness).";
    case GL_TABLE_TOO_LARGE1: return "Part of the ARB_imaging extension1. 1 These error codes are deprecated in 3.0 and removed in 3.1 core and above.";
    default: return "";
    }
}

static void CheckOpenGL(const char* CallStr) {
    GLenum Error = glGetError();
    if (Error != GL_NO_ERROR) {
        fprintf(stderr, "%s\n", CallStr);

        int I = 0;
        do {
            fprintf(stderr, "\t[%d]: %s (%d) %s\n", I++, ErrorStringOpenGL(Error), Error, ErrorDescriptionOpenGL(Error));
        } while ((Error = glGetError()) != GL_NO_ERROR);
    }
}

// NOTE: Do NOT call this function as part of a single statement if, while, etc.
#define glCheck(X) X; do { CheckOpenGL(#X); } while(0)

typedef Eigen::Vector2f v2;
typedef Eigen::Vector3f v3;
typedef Eigen::Vector4f v4;
typedef Eigen::Matrix2f m2;
typedef Eigen::Matrix3f m3;
typedef Eigen::Matrix4f m4;

static m2 Rotation(float Angle) {
    v2 X(cosf(Angle), sinf(Angle));
    v2 Y(-sinf(Angle), cosf(Angle));

    m2 R;
    R << X, Y;
    return R;
}

static m3 RotationX(float Angle) {
    float C = cosf(Angle);
    float S = sinf(Angle);
    v3 X(1.0f, 0.0f, 0.0f);
    v3 Y(0.0f, C, S);
    v3 Z(0.0f, -S, C);

    m3 R;
    R << X, Y, Z;
    return R;
}

static m3 RotationY(float Angle) {
    float C = cosf(Angle);
    float S = sinf(Angle);
    v3 X(C, 0.0f, -S);
    v3 Y(0.0f, 1.0f, 0.0f);
    v3 Z(S, 0.0f, C);

    m3 R;
    R << X, Y, Z;
    return R;
}

static m3 RotationZ(float Angle) {
    float C = cosf(Angle);
    float S = sinf(Angle);
    v3 X(C, S, 0.0f);
    v3 Y(-S, C, 0.0f);
    v3 Z(0.0f, 0.0f, 1.0f);

    m3 R;
    R << X, Y, Z;
    return R;
}

static m4 M4(m3 M, v3 T = v3(0.0, 0.0, 0.0)) {
    v4 X(1.0f, 0.0f, 0.0f, 0.0f);
    v4 Y(0.0f, 1.0f, 0.0f, 0.0f);
    v4 Z(0.0f, 0.0f, 1.0f, 0.0f);
    v4 W(T[0], T[1], T[2], 1.0f);

    m4 Result;
    Result << X, Y, Z, W;

    Result(0, 0) = M(0, 0);
    Result(0, 1) = M(0, 1);
    Result(0, 2) = M(0, 2);
    Result(1, 0) = M(1, 0);
    Result(1, 1) = M(1, 1);
    Result(1, 2) = M(1, 2);
    Result(2, 0) = M(2, 0);
    Result(2, 1) = M(2, 1);
    Result(2, 2) = M(2, 2);
    return Result;
}

static m4 Translation(v3 T) {
    m4 Result = m4::Identity();
    Result(0, 3) = T[0];
    Result(1, 3) = T[1];
    Result(2, 3) = T[2];
    return Result;
}

static m3 Scale(float S) {
    m3 Result = S * m3::Identity();
    return Result;
}

static m4 Perspective(float FoVY, float WoH, float N, float F) {
    m4 Result;

    float T = tanf(FoVY / 2.0f);
    float SX = 1.0f / (T * WoH);
    float SY = 1.0f / T;
    float A = (N + F) / (N - F);
    float B = 2.0f * N * F / (N - F);

    Result <<
        SX, 0.0f, 0.0f, 0.0f,
        0.0f, SY, 0.0f, 0.0f,
        0.0f, 0.0f, A, B,
        0.0f, 0.0f, -1.0f, 0.0f;
    return Result;
}

struct vertex {
    v3 Position;
    v3 Color;
    v3 ShaderNormal;
    v2 TextCoords;
};

static char* ReadFile(const char* FileName) {
    FILE* File = fopen(FileName, "r");
    fseek(File, 0, SEEK_END);
    long int FileSize = ftell(File);
    fseek(File, 0, SEEK_SET);

    char* Bytes = (char*)malloc(FileSize + 1);
    Bytes[FileSize] = 0;
    fread(Bytes, FileSize, 1, File);
    return Bytes;
}

//only use lighting no model
// utility function for loading a 2D texture from file
// ---------------------------------------------------
//unsigned int loadTexture(const char* path);




unsigned int loadTexture(char const* path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}








//GLuint cube_vertices;
/*static GLuint cube_vertices[] = {
    //front
    -1.0, -1.0,  1.0,
    1.0, -1.0,  1.0,
    1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    // back
    -1.0, -1.0, -1.0,
    1.0, -1.0, -1.0,
    1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
}; 
*/
GLuint cube_colours[] = {
    /*front colors*/
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
    /* back colors*/
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 1.0, 1.0,
};
static unsigned int cube_elements_indices[] = {
    /*front*/
    0, 1, 2,
    2, 3, 0,
    /*top*/
    1, 5, 6,
    6, 2, 1,
    /*back*/
    7, 6, 5,
    5, 4, 7,
    /*bottom*/
    4, 0, 3,
    3, 7, 4,
    /*left*/
    4, 5, 1,
    1, 0, 4,
    /*right*/
    3, 2, 6,
    6, 7, 3
};
class softBodyEdge {
public:
    unsigned int PositionIndex0;
    unsigned int PositionIndex1;
    //unsigned int TriangleIndex0;
    //unsigned int TriangleIndex1;

    float lambdaLength;
    float InitialLength;
    //float InitialAngle;

    float length;
    //float angle;
    softBodyEdge(unsigned int PositionIndex0, unsigned int PositionIndex1, float InitialLength, float length) {
        this->PositionIndex0 = PositionIndex0;
        this->PositionIndex1 = PositionIndex1;
        this->lambdaLength = 0;
        this->InitialLength = InitialLength;
        this->length = length;
    }
};

static std::vector <softBodyEdge> Edges = {
   
    softBodyEdge(0,3,1.0f,1.0f),
    //softBodyEdge(3,0,1.0f,1.0f),
    softBodyEdge(0,4,1.0f,1.0f),
    //softBodyEdge(4,0,1.0f,1.0f),
    softBodyEdge(0,1,1.0f,1.0f),
    //softBodyEdge(1,0,1.0f,1.0f),

    softBodyEdge(1,2,1.0f,1.0f),
    //softBodyEdge(2,1,1.0f,1.0f),
    softBodyEdge(1,5,1.0f,1.0f),
    //softBodyEdge(5,1,1.0f,1.0f),

    softBodyEdge(2,3,1.0f,1.0f),
    //softBodyEdge(3,2,1.0f,1.0f),
    softBodyEdge(2,6,1.0f,1.0f),
    //softBodyEdge(6,2,1.0f,1.0f),

    softBodyEdge(3,7,1.0f,1.0f),
    //softBodyEdge(7,3,1.0f,1.0f),

    softBodyEdge(4,7,1.0f,1.0f),
    //softBodyEdge(7,4,1.0f,1.0f),
    softBodyEdge(4,5,1.0f,1.0f),
    //softBodyEdge(5,4,1.0f,1.0f),

    softBodyEdge(5,6,1.0f,1.0f),
    //softBodyEdge(6,5,1.0f,1.0f),

    softBodyEdge(6,7,1.0f,1.0f),
    //softBodyEdge(7,6,1.0f,1.0f),


     softBodyEdge(0,2,1.41f,1.41f),
     softBodyEdge(0,5,1.41f,1.41f),
     softBodyEdge(0,7,1.41f,1.41f),

     softBodyEdge(1,6,1.41f,1.41f),

     softBodyEdge(4,6,1.41f,1.41f),

     softBodyEdge(6,3,1.41f,1.41f),



     softBodyEdge(4,2,1.73f,1.73f),
     softBodyEdge(5,3,1.73f,1.73f),


};
class model {
public:
    //softBodyTriangle Triangles[12];
    
    float initialVolume, volume;

    vertex Points[8] = {
{   { -0.5f,  -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { 0.5f,  -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { 0.5f, 0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { -0.5f,  0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { -0.5f, -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { 0.5f, -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { 0.5f,  0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
{   { -0.5f,  0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    };
    //;
    float masses[8] = { 2,2,2,2,2,2,2,2 };
    v3 velocities[8] = { {0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0} };
    v3 acceleration[8] = { {0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0} };
    v3 forces[8] = { {0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0} };
    float w[8] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, };


};
static void collisionConstraint(v3& x0, v3 x1, v3 N, float invM, float& lambdaN, float& lambdaT, float MuS, v3 p0p) {
    float C = (x0 - x1).dot(N);
    if (C < 0) {
        float deltaLambdaN = -C / invM;
        v3 deltaX = -C * N;

        v3 deltaP = x0 - p0p;
        v3 deltapt = deltaP - deltaP.dot(N) * N;
        float deltaLambdaT = (deltapt / invM).norm();

        lambdaT += deltaLambdaT;
        lambdaN += deltaLambdaN;

        if (lambdaT < MuS * lambdaN) {
            deltaX -= deltapt;
        }

        x0 += deltaX;   

    }

}

static void distanceConstraint(v3& x0, v3& x1, float & lambda, float invM0, float invM1, float InitialLength, float alpha) {
    
    v3 diff = x0 - x1;
    float dist = diff.norm();
    v3 dir;
        if (dist  == 0) {
            dir = v3(0.0, 1.0, 0.0);

        }
        else {
            dir = diff / dist;
        }
        float Cj = dist - InitialLength;
        if (Cj != 0) {
            float deltaLambda = (-Cj - alpha *(lambda)) / (invM0 + invM1 + alpha);
            x0 = x0 + ((invM0 * deltaLambda) * dir);
            x1 = x1 - ((invM0 * deltaLambda) * dir);
            lambda += deltaLambda;
            
        }
}
//distanceConstraint(Position[0],Position[1],,1.0,1.0,1.0,1000)
class softBodyTriangle {
public:
    unsigned int PositionIndices[3];


};


/*
static vertex MeshData[] = {
    //cube
    
    //cube
    //z axis positive cube face
    { { 0.5f,  0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { -0.5f,  0.5f, 0.5f }, { 1.0f, 1.0f, 0.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f } },

    { { 0.5f,  0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f } },

    //z axis negative cube face
    { { 0.5f,  0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { -0.5f,  0.5f, -0.5f }, { 1.0f, 1.0f, 0.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    { { 0.5f,  0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    //x axis positive cube face
    { { 0.5f,  0.5f,  0.5f },  { 0.0f, 0.0f, 1.0f } },
    { { 0.5f,  0.5f, -0.5f }, { 0.0f, 1.0f, 1.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    { { 0.5f,  0.5f,  0.5f },  { 0.0f, 0.0f, 1.0f } },
    { { 0.5f,  -0.5f, 0.5f }, { 0.0f, 1.0f, 1.0f } },
    { { 0.5f,  -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    //x axis negative cube face
    { { -0.5f,  0.5f,  0.5f },  { 0.0f, 0.0f, 1.0f } },
    { { -0.5f,  0.5f, -0.5f }, { 0.0f, 1.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    { { -0.5f,  0.5f,  0.5f },  { 0.0f, 0.0f, 1.0f } },
    { { -0.5f,  -0.5f, 0.5f }, { 0.0f, 1.0f, 1.0f } },
    { { -0.5f,  -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

    //y axis positive cube face
   { { 0.5f,  0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
   { { -0.5f, 0.5f, 0.5f }, { 1.0f, 1.0f, 0.0f } },
   { { -0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

   { { 0.5f,  0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
   { { 0.5f,  0.5f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
   { { -0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },


   //y axis negative cube face
   { { 0.5f,  -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
   { { -0.5f, -0.5f, 0.5f }, { 1.0f, 1.0f, 0.0f } },
   { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

   { { 0.5f,  -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
   { { 0.5f,  -0.5f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
   { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
    

    
}; 
static void convertPoints() {
    for (int i = 0; i < 36; ++i) {
        MeshData[i].Position[0] = cube_elements_indices[i];
    }

}
*/
class objectFace {
public:
    unsigned int PositionIndex0;
    unsigned int PositionIndexA;
    unsigned int PositionIndexB;
    v3 normal;
    objectFace(unsigned int PositionIndex0, unsigned int PositionIndexA, unsigned int PositionIndexB) {
        this->PositionIndex0 = PositionIndex0;
        this->PositionIndexA = PositionIndexA;
        this->PositionIndexB = PositionIndexB;

    }


};
static std::vector <objectFace> PlaneFaces = {
    objectFace(2,1,4)

};

static std::vector <objectFace> StaticCubeFaces = {
    objectFace(32,31,34), //y axis positive cube face
    objectFace(38,40,37), //y axis negative cube face

    objectFace(8,10,7), //z axis positive cube face
    objectFace(14,13,16), //z axis negative cube face

    objectFace(20,19,22), //x axis positive cube face
    objectFace(26,28,25), //x axis negative cube face

    
};

class collision {
public:
    int index;
    v3 normal;
    v3 point;
    float lambdaN;
    float lambdaT;

};
static vertex plane[] = {

    //plane
     { { 5.0f,  0.0f, 5.0f }, { 1.0f, 0.0f, 0.0f } },
    { { -5.0f, 0.0f, 5.0f }, { 1.0f, 1.0f, 0.0f } },
    { { -5.0f, 0.0f, -5.0f }, { 0.0f, 1.0f, 0.0f } },

    { { 5.0f,  0.0f, 5.0f }, { 1.0f, 0.0f, 0.0f } },
    { { 5.0f, 0.0f, -5.0f }, { 0.0f, 0.0f, 1.0f } },
    { { -5.0f, 0.0f, -5.0f }, { 0.0f, 1.0f, 0.0f } },

    //cube
 //z axis positive cube face
 { { 5.5f,  2.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } }, //6
 { { 4.5f,  2.0f, 2.5f }, { 1.0f, 1.0f, 0.0f } },
 { { 4.5f,  -1.0f, 2.5f }, { 0.0f, 1.0f, 0.0f } },

 { { 5.5f,  2.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } },
 { { 5.5f,  -1.0f, 2.5f }, { 0.0f, 0.0f, 1.0f } },
 { { 4.5f,  -1.0f, 2.5f }, { 0.0f, 1.0f, 0.0f } },

 //z axis negative cube face
 { { 5.5f,  2.0f, -0.5f }, { 1.0f, 0.0f, 0.0f } }, //12
 { { 4.5f,  2.0f, -0.5f }, { 1.0f, 1.0f, 0.0f } },
 { { 4.5f,  -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 { { 5.5f,  2.0f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
 { { 5.5f,  -1.0f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
 { { 4.5f,  -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 //x axis positive cube face
 { { 5.5f,  2.0f,  2.5f },  { 0.0f, 0.0f, 1.0f } }, //18
 { { 5.5f,  2.0f, -0.5f }, { 0.0f, 1.0f, 1.0f } },
 { { 5.5f,  -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 { { 5.5f,  2.0f,  2.5f },  { 0.0f, 0.0f, 1.0f } }, //21
 { { 5.5f,   -1.0f, 2.5f }, { 0.0f, 1.0f, 1.0f } },
 { { 5.5f,   -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 //x axis negative cube face
 { { 4.5f,  2.0f,  2.5f },  { 0.0f, 0.0f, 1.0f } }, //24
 { { 4.5f,  2.0f, -0.5f }, { 0.0f, 1.0f, 1.0f } },
 { { 4.5f,  -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 { { 4.5f,  2.0f,  2.5f },  { 0.0f, 0.0f, 1.0f } }, //27
 { { 4.5f,   -1.0f, 2.5f }, { 0.0f, 1.0f, 1.0f } },
 { { 4.5f,   -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

 //y axis positive cube face
{ {  5.5f,  2.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } }, //30
{ { 4.5f,   2.0f, 2.5f }, { 1.0f, 1.0f, 0.0f } },
{ { 4.5f,  2.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

{ {  5.5f,  2.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } }, //33
{ {  5.5f,  2.0f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
{ { 4.5f,   2.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },


//y axis negative cube face
{ {  5.5f,  -1.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } }, //36
{ { 4.5f,   -1.0f, 2.5f }, { 1.0f, 1.0f, 0.0f } },
{ { 4.5f,   -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

{ {  5.5f,   -1.0f, 2.5f }, { 1.0f, 0.0f, 0.0f } }, //39
{ {  5.5f,   -1.0f, -0.5f }, { 0.0f, 0.0f, 1.0f } },
{ { 4.5f,    -1.0f, -0.5f }, { 0.0f, 1.0f, 0.0f } },

};

static void scanForCollisionsWithSurfaces(v3 x0, bool& colFound, collision& col, static vertex plane[], static std::vector <objectFace> StaticCubeFaces) {
    /*
    colFound = true;
    for (int i = 0; i < PlaneFaces.size() && colFound; i++) {
        v3 a = plane[PlaneFaces[i].PositionIndexA].Position - plane[PlaneFaces[i].PositionIndex0].Position;
        v3 b = plane[PlaneFaces[i].PositionIndexB].Position - plane[PlaneFaces[i].PositionIndex0].Position;
        v3 cross = a.cross(b);
        cross.normalize();
        //std::cout << "plane cross product " << cross << std::endl;
        PlaneFaces[i].normal = cross;
        if ((plane[PlaneFaces[i].PositionIndexA].Position - (x0 + 0.005 * PlaneFaces[i].normal)).norm() > (plane[PlaneFaces[i].PositionIndexA].Position - x0).norm()) {
            //distance between point x0 and a point of plane face
            colFound = false;
        }
    }
    for (int i = 0; i < PlaneFaces.size() && colFound; i++) {
        v3 v = x0 - plane[PlaneFaces[i].PositionIndexA].Position;
        float dist = v.dot(-PlaneFaces[i].normal);//normal has to be negative since we're looking inwards
        //dot product used to accurately find distance for use in cases with more complex normal vectors
        //if this is smallest distance it will be the collision point {
        
        col.point = x0 - dist * -PlaneFaces[i].normal;//normal has to be negative since we're looking inwards
        col.normal = PlaneFaces[i].normal;
        std::cout << "object collision point " << col.point << std::endl;
    }
    */
   
    for (int i = 0; i < StaticCubeFaces.size() && colFound; i++) {
        v3 a = plane[StaticCubeFaces[i].PositionIndexA].Position - plane[StaticCubeFaces[i].PositionIndex0].Position;
        v3 b = plane[StaticCubeFaces[i].PositionIndexB].Position - plane[StaticCubeFaces[i].PositionIndex0].Position;
        v3 cross = a.cross(b);
        cross.normalize();
        //std::cout << "object face cross product " << cross << std::endl;
        StaticCubeFaces[i].normal = cross;
        if ((plane[StaticCubeFaces[i].PositionIndexA].Position - (x0 + 0.005 * StaticCubeFaces[i].normal)).norm() > (plane[StaticCubeFaces[i].PositionIndexA].Position - x0).norm()) {
            //distance between point x0 and a point of cube face
            colFound = false;
        }
    }
    std::vector <float> distances;
    int minIndex = 0;
    float minDist = 0.0;
    for (int i = 0; i < StaticCubeFaces.size() && colFound; i++) {
        //std::cout << "COLLISION FOUND!";
        v3 v = x0 - plane[StaticCubeFaces[i].PositionIndexA].Position;
        float dist = v.dot(-StaticCubeFaces[i].normal);//normal has to be negative since we're looking inwards
        //dot product used to accurately find distance for use in cases with more complex normal vectors
        distances.push_back(dist);
        std::sort(distances.begin(), distances.end());
        if (dist == distances[0]) {
            minIndex = i;
            minDist = dist;
        }
    }
    if (colFound) {
        col.point = x0 - minDist * -StaticCubeFaces[minIndex].normal;//normal has to be negative since we're looking inwards
        col.normal = StaticCubeFaces[minIndex].normal;
        std::cout << " distance,normal :( " << minDist <<", "<< StaticCubeFaces[minIndex].normal << " )" << std::endl;
        //std::cout << "object collision point " << col.point << std::endl;
    }
    // negp1 (-1,0,0) negp2(0,-1,0) negp3(0,0,-1)
    // negp1negp2 (1,-1, 0) 
    // negp1negp3 (1, 0,-1) 
    // vector perpendicular to plane = negp1negp3 x negp1negp2 = ( -1, -1, -1)

    // p1 (1,0,0) p2(0,1,0) p3(0,0,1)
    // vector p1p2 = (-1, 1, 0)
    // vector p1p3 = (-1, 0, 1)
    // vector perpendicular to plane = p1p2 x p1p3 = ( 1, 1, 1)
    // normalised perpendicular vector = (1/sqrt(3), 1/sqrt(3), 1/sqrt(3) )


   
}

static int CompileShader(GLuint* Shader, const char* VertexShader, const char* FragmentShader) {
    int Success0;

    GLuint VID = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(VID, 1, (const GLchar* const*)&VertexShader, 0);
    glCompileShader(VID);

    glGetShaderiv(VID, GL_COMPILE_STATUS, &Success0);
    if (Success0 == GL_FALSE) {
        GLchar Buf[1024] = { 0 };
        GLsizei Length = 0;
        glGetShaderInfoLog(VID, ArrayCount(Buf), &Length, Buf);
        fprintf(stderr, "%.*s", Length, Buf);
        return 1;
    }

    GLuint FID = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(FID, 1, (const GLchar* const*)&FragmentShader, 0);
    glCompileShader(FID);

    glGetShaderiv(FID, GL_COMPILE_STATUS, &Success0);
    if (Success0 == GL_FALSE) {
        GLchar Buf[1024] = { 0 };
        GLsizei Length = 0;
        glGetShaderInfoLog(FID, ArrayCount(Buf), &Length, Buf);
        fprintf(stderr, "%.*s", Length, Buf);
        return 1;
    }

    GLuint Program = glCreateProgram();
    glAttachShader(Program, VID);
    glAttachShader(Program, FID);
    glLinkProgram(Program);

    glGetProgramiv(Program, GL_LINK_STATUS, &Success0);
    if (Success0 == GL_FALSE) {
        GLchar Buf[1024] = { 0 };
        GLsizei Length = 0;
        glGetProgramInfoLog(Program, ArrayCount(Buf), &Length, Buf);
        fprintf(stderr, "%.*s", Length, Buf);
        return 1;
    }

    glDetachShader(Program, VID);
    glDetachShader(Program, FID);
    glDeleteShader(VID);
    glDeleteShader(FID);

    *Shader = Program;
    return 0;
}

static int LoadShader(GLuint* Shader, const char* VertexPath, const char* FragmentPath) {
    char* VertexCode = ReadFile(VertexPath);
    char* FragmentCode = ReadFile(FragmentPath);

    int Error = CompileShader(Shader, VertexCode, FragmentCode);

    free(VertexCode);
    free(FragmentCode);

    return Error;
}

typedef struct {
    Sint16 L;
    Sint16 R;
} audio_sample;
static float Min(float a, float b) {
    if (a < b)
        return a;
    else if (b < a)
        return b;
}
typedef struct {
    int Freq;
    long SamplesDone;
} audio_data;

static void MyAudioCallback(void* Userdata, Uint8* Stream, int Len) {
    audio_data* Audio = (audio_data*)Userdata;
    int SampleCount = Len / sizeof(audio_sample);

    audio_sample* Samples = (audio_sample*)Stream;
    for (int I = 0; I < SampleCount; ++I) {

        double Time = (Audio->SamplesDone + I) / (double)(Audio->Freq);

        double ValueF = sin(264 * 2 * M_PI * Time);
        Sint16 Value = 1000 * ValueF;

        Samples[I].L = Value;
        Samples[I].R = Value;
    }

    Audio->SamplesDone += SampleCount;
}

int main(int, char**) {


    //changing cube pos
    //for (int i = 6; i < 42; i++) {
    //    plane[i].Position(0) -= 5.2;
    //    plane[i].Position(1) -= 3.0;

    //}
    model Body;
    
    
    if (SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return -1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);

    int WindowWidth = 192 * 6;
    int WindowHeight = 108 * 6;
    SDL_Window* Window = SDL_CreateWindow("SDL Template", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WindowWidth, WindowHeight, SDL_WINDOW_OPENGL);
    if (Window == 0) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return -1;
    }

    SDL_GLContext Context = SDL_GL_CreateContext(Window);
    if (Context == 0) {
        fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError());
        return -1;
    }

    if (SDL_GL_SetSwapInterval(1)) {
        fprintf(stderr, "SDL_GL_SetSwapInterval: %s\n", SDL_GetError());
    }

    if (gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress) == 0) {
        fprintf(stderr, "glad gl load failed\n");
        return -1;
    }

    //original
    GLuint Shader;
    if (LoadShader(&Shader, "default.vs", "default.fs")) {
        fprintf(stderr, "shader compilation failed\n");
        return -1;
    }



    // build and compile shaders learnopengl------------------
    // -------------------------
    Sh sh("advanced_lighting.vs", "advanced_lighting.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
    float planeVertices[] = {
        // positions            // normals         // texcoords
         10.0f, -2.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
        -10.0f, -2.5f,  10.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
        -10.0f, -2.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,

         10.0f, -2.5f,  10.0f,  0.0f, 1.0f, 0.0f,  10.0f,  0.0f,
        -10.0f, -2.5f, -10.0f,  0.0f, 1.0f, 0.0f,   0.0f, 10.0f,
         10.0f, -2.5f, -10.0f,  0.0f, 1.0f, 0.0f,  10.0f, 10.0f
    };


    //learnopengl--------------------------------------------------------------------------------
    // plane VAO
    unsigned int planeVAO, planeVBO;
    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glBindVertexArray(0);
    //learnopengl--------------------------------------------------------------------------------






    GLuint VAO2, VBO2;
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(plane), plane, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    

    
    GLuint VAO,  VBO;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Body.Points), Body.Points, GL_DYNAMIC_DRAW);
    //new ibo

    unsigned int EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_elements_indices), cube_elements_indices, GL_DYNAMIC_DRAW);


 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);



    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
  



    glCheck(glEnable(GL_FRAMEBUFFER_SRGB));
    glEnable(GL_DEPTH_TEST);

   
    // configure global opengl state
    // -----------------------------
    //glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   
    // load textures
    // -------------
    unsigned int floorTexture = loadTexture("C:/Users/User/source/repos/opengl/opengl/wood.png");


   




    GLfloat ModelPosition[2] = { 0 };

    float DeltaTime = 0.0;
    Uint32 LastTicks = SDL_GetTicks();
    float Angle = 0.0f;

    float LastMouseX = 0.0f;
    float LastMouseY = 0.0f;
    float MouseX = 0.0f;
    float MouseY = 0.0f;

    float AngleX = 0.0f;
    float AngleY = 0.0f;

    int IsButtonDown = 0;
    float Zoom = 0.4f;

    int Running = 1;

    
    
    while (Running) {
        
        SDL_Event Event;
        LastMouseX = MouseX;
        LastMouseY = MouseY;
        while (SDL_PollEvent(&Event)) {
            switch (Event.type) {
            case SDL_QUIT:
                Running = 0;
                break;
            case SDL_MOUSEBUTTONDOWN:
                ModelPosition[0] = (2.0f * Event.button.x) / WindowWidth - 1.0f;
                ModelPosition[1] = (-2.0f * Event.button.y) / WindowHeight + 1.0f;
                IsButtonDown = 1;
                break;
            case SDL_MOUSEBUTTONUP:
                IsButtonDown = 0;
                break;
            case SDL_MOUSEMOTION:
                MouseX = Event.motion.x;
                MouseY = Event.motion.y;
                break;


                /* Look for a keypress */
            case SDL_KEYDOWN:
                switch (Event.key.keysym.sym) {

                case SDLK_SPACE:
                    for (int i = 0; i < 8; i++) {
                        Body.velocities[i](1) += 5;

                    }
                    Zoom *= 1.1f;
                    break;
                case SDLK_LEFT:
                    for (int i = 0; i < 8; i++) {
                        Body.velocities[i](0) -=0.5;

                    }
                    Zoom *= 1.1f;
                    break;
                case SDLK_RIGHT:
                    for (int i = 0; i < 8; i++) {
                        Body.velocities[i](0) += 0.5;

                    }
                    Zoom *= 1.1f;
                    break;
                case SDLK_UP:
                    for (int i = 0; i < 8; i++) {
                        Body.velocities[i](2) -= 0.5;

                    }
                    Zoom *= 1.1f;
                    break;
                case SDLK_DOWN:
                    for (int i = 0; i < 8; i++) {
                        Body.velocities[i](2) += 0.5;

                    }
                    Zoom *= 1.1f;
                    break;


                default:
                    break;
                }
               





            case SDL_MOUSEWHEEL:
                if (Event.wheel.y < 0) {
                    Zoom *= 1.1f;
                }
                else if (Event.wheel.y > 0) {
                    Zoom /= 1.1f;
                }
                break;
            default:
                break;
            }
        }

        float DeltaMouseX = MouseX - LastMouseX;
        float DeltaMouseY = MouseY - LastMouseY;

        if (IsButtonDown) {
            AngleX -= 0.004f * DeltaMouseY;
            AngleY -= 0.004f * DeltaMouseX;
        }


        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glCheck(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

        Angle += DeltaTime;
        m3 R = RotationX(AngleX) * RotationY(AngleY);
        m4 R4 = M4(Scale(0.6));//M4(R, v3(0.1, 0.0, 0.0));

        float CameraRadius = 6.0f;
        m4 C = M4(Scale(Zoom)) * M4(RotationY(AngleY)) * M4(RotationX(AngleX)) * Translation(v3(0.0f, 0.0f, CameraRadius));
        m4 V = Translation(v3(0.0f, 0.0f, -CameraRadius)) * M4(RotationX(-AngleX)) * M4(RotationY(-AngleY)) * M4(Scale(1.0f / Zoom));
        m4 P = Perspective(1.39626, WindowWidth / (float)WindowHeight, 0.1f, 100.0f);

       


        glCheck(glUseProgram(Shader));
        //glCheck(glUniform2fv(glGetUniformLocation(Shader, "ModelP"), 1, ModelPosition));
        glCheck(glUniformMatrix4fv(glGetUniformLocation(Shader, "R"), 1, GL_FALSE, R4.data()));
        glCheck(glUniformMatrix4fv(glGetUniformLocation(Shader, "MatV"), 1, GL_FALSE, V.data()));
        glCheck(glUniformMatrix4fv(glGetUniformLocation(Shader, "MatP"), 1, GL_FALSE, P.data()));

        //learnopengl----------------------------------------------------
        // floor
        glBindVertexArray(planeVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, floorTexture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        //---------------------------------------------------------------


        //update positions with the already binded VBO and VAO
        glGenVertexArrays(1, &VAO);
        glCheck(glBindVertexArray(VAO));
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Body.Points), Body.Points, GL_DYNAMIC_DRAW);
        //new ibo

        unsigned int EBO;
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_elements_indices), cube_elements_indices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(0));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        
        glCheck(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0));
        //36 overlapping cube edge points stored as indices, 6 per face

        


        glCheck(glBindVertexArray(VAO2));
        glCheck(glDrawArrays(GL_TRIANGLES, 0, 42));
        //6 edges points of plane -> 42 edges points of plane + cube
        
        

        glCheck(glBindVertexArray(0));

        glCheck(glUseProgram(0));

        SDL_GL_SwapWindow(Window);

        Uint32 CurrTicks = SDL_GetTicks();
        //DeltaTime = ((CurrTicks - LastTicks) / 1000.0f);
        DeltaTime = 0.01;
        //std::cout << DeltaTime << std::endl;
        LastTicks = CurrTicks;
        std::vector <v3> oldpos;
        std::vector <collision> collisions;
        for (int i = 0; i < 8; i++) {
            v3 nextVelocity = Body.velocities[i];
            nextVelocity(1) = Body.velocities[i](1) + DeltaTime * -10;
            oldpos.push_back(Body.Points[i].Position);
            Body.Points[i].Position += nextVelocity * DeltaTime;

        }
        for (int i = 0; i < Edges.size(); i++) {
            Edges[i].lambdaLength = 0.0;
        }

        for (int i = 0; i < 8; i++) {//maybe add points in the middle of edges for the cube, so we cant enter an object with the edge or add collision detection for vertices of static objects either works
            bool colFound = true;
            collision Col;
             scanForCollisionsWithSurfaces(Body.Points[i].Position, colFound, Col, plane, PlaneFaces);
            if (colFound) {
                Col.index = i;
                //Col.point = Body.Points[i].Position;
                //Col.point(1) = 0;
                //Col.normal = v3(0, 1.0 ,0);
                //test for normalalized unit vector
                //normal vector for the plane, gonna need different normal vectors for other objects
                Col.lambdaN = 0;
                Col.lambdaT = 0;
                collisions.push_back(Col);

                //Body.Points[i].Position(1) = 0;

            } 
            //if (!colFound) {

                colFound = true;
                scanForCollisionsWithSurfaces(Body.Points[i].Position, colFound, Col, plane, StaticCubeFaces);
                if (colFound) {
                    std::cout << " TRUE";
                    Col.index = i;
                    Col.lambdaN = 0;
                    Col.lambdaT = 0;
                    collisions.push_back(Col);
                }
            //}
        }
        for (int j = 0; j < 20; j++) {
            for (int i = 0; i < Edges.size(); i++) {

                
                
                distanceConstraint(Body.Points[Edges[i].PositionIndex0].Position, Body.Points[Edges[i].PositionIndex1].Position, Edges[i].lambdaLength, 1.0, 1.0, Edges[i].InitialLength, 10);
               
            }
            std::cout << "number of collisions: " << collisions.size() << std::endl;
            for (int i = 0; i < collisions.size(); i++) {
                //collision c = collisions[i];
                collisionConstraint(Body.Points[collisions[i].index].Position, collisions[i].point, collisions[i].normal, 1.0, collisions[i].lambdaN, collisions[i].lambdaT, 0.5, oldpos[i]);
            }
            
        }
       
        
        
        for (int i = 0; i < 8; i++) {
            Body.velocities[i] = (Body.Points[i].Position - oldpos[i]) / DeltaTime;
            //std::cout <<"x: " << Body.Points[i].Position(0) << "y: " << Body.Points[i].Position(1) << "z: " << Body.Points[i].Position(2) << std::endl;
        }

       for (int i = 0; i < collisions.size(); i++) {
           collision c = collisions[i];
            v3 velocityN = Body.velocities[c.index].dot(c.normal) * c.normal;
           v3 velocityT = Body.velocities[c.index] - velocityN;
            v3 deltaV = -Min(0.1 * abs(c.lambdaN / DeltaTime) * 1.0, velocityT.norm()) * velocityT.normalized();
           Body.velocities[c.index] += deltaV;
        }
       
        
    }
   
    SDL_GL_DeleteContext(Context);
    SDL_DestroyWindow(Window);
    SDL_Quit();

    return 0;
}
