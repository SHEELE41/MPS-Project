#include <jni.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <android/log.h>
#include <android/bitmap.h>

#include <CL/opencl.h>

#define LOG_TAG "DEBUG"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

#define CL_FILE "/data/local/tmp/mirror.cl"

#define checkCL(expression) {                        \
    cl_int err = (expression);                       \
    if (err < 0 && err > -64) {                      \
        LOGD("Error on line %d. error code: %d\n",   \
                __LINE__, err);                      \
        exit(0);                                     \
    }                                                \
}

int fd = 0;

JNIEXPORT jint JNICALL
Java_com_parlab_smarthome_driver_JNIDriver_openDriver(JNIEnv *env, jclass clazz, jstring path) {
    jboolean iscopy;
    const char *path_utf = (*env)->GetStringUTFChars(env, path, &iscopy);
    fd = open(path_utf, O_RDONLY);
    (*env)->ReleaseStringUTFChars(env, path, path_utf);

    if (fd < 0) return -1;
    else return 1;
}

JNIEXPORT void JNICALL
Java_com_parlab_smarthome_driver_JNIDriver_closeDriver(JNIEnv *env, jclass clazz) {
    if(fd>0) {
        close(fd);
    }
}

JNIEXPORT jchar JNICALL
Java_com_parlab_smarthome_driver_JNIDriver_readDriver(JNIEnv *env, jobject thiz) {
    char ch = 0;

    if (fd > 0) {
        read(fd, &ch, 1);
    }

    return ch;
}

JNIEXPORT jint JNICALL
Java_com_parlab_smarthome_driver_JNIDriver_getInterrupt(JNIEnv *env, jobject thiz) {
    int ret = 0;
    char value[100];
    char *ch1 = "Up";
    char *ch2 = "Down";
    char *ch3 = "Left";
    char *ch4 = "Right";
    char *ch5 = "Center";
    ret = read(fd, &value, 100);

    if (ret < 0)
        return -1;
    else {
        if (strcmp(ch1, value) == 0)
            return 1;
        else if (strcmp(ch2, value) == 0)
            return 2;
        else if (strcmp(ch3, value) == 0)
            return 3;
        else if (strcmp(ch4, value) == 0)
            return 4;
        else if (strcmp(ch5, value) == 0)
            return 5;
    }
    return 0;
}

JNIEXPORT jobject JNICALL
Java_com_parlab_smarthome_driver_JNIDriver_mirrorGPU(JNIEnv *env, jobject thiz, jobject bitmap) {
    LOGD("reading bitmap info...");
    AndroidBitmapInfo info;
    int ret;
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return NULL;
    }
    LOGD("width:%d height:%d stride:%d", info.width, info.height, info.stride);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format is not RGBA_8888!");
        return NULL;
    }

    //read pixels of bitmap into native memory :
    LOGD("reading bitmap pixels...");
    void *bitmapPixels;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return NULL;
    }

    unsigned char *src = (unsigned char *) bitmapPixels;
    unsigned char *tempPixels = (unsigned char *) malloc(info.height * info.width * 4);
    int pixelsCount = info.height * info.width * 4;
    memcpy(tempPixels, src, sizeof(unsigned char) * pixelsCount);

    FILE *file_handle;
    char *kernel_file_buffer, *file_log;
    size_t kernel_file_size, log_size;

    unsigned char *cl_file_name = CL_FILE;
    unsigned char *kernel_name = "kernel_blur";

    // Device input buffers
    cl_mem d_src;
    // Device output buffer
    cl_mem d_dst;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    file_handle = fopen(CL_FILE, "r");
    if (file_handle == NULL) {
        printf("Couldn't find the file");
        exit(1);
    }

    //read kernel file
    fseek(file_handle, 0, SEEK_END);
    kernel_file_size = ftell(file_handle);
    rewind(file_handle);
    kernel_file_buffer = (char *) malloc(kernel_file_size + 1);
    kernel_file_buffer[kernel_file_size] = '\0';
    fread(kernel_file_buffer, sizeof(char), kernel_file_size, file_handle);
    fclose(file_handle);

    // Initialize vectors on host
    int i;

    size_t globalSize, localSize, grid;

    // Number of work items in each local work group
    localSize = 64;
    int n_pix = info.width * info.height;

    // Number of total work items - localSize must be devisor
    grid = ((n_pix) % localSize) ? ((n_pix) / localSize) + 1 : (n_pix) / localSize;
    globalSize = grid * localSize;

    cl_int err;

    // Bind to platform
    checkCL(clGetPlatformIDs(1, &cpPlatform, NULL));

    // Get ID for the device
    checkCL(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    checkCL(err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernel_file_buffer, &kernel_file_size, &err);
    checkCL(err);

    // Build the program executable
    checkCL(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    LOGD("error 22 check");
    checkCL(err);
    if (err != CL_SUCCESS) {
        LOGD("%s", err);
        size_t len;
        char buffer[4096];
        LOGD("Error: Failed to build program executable!");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                              buffer, &len);

        LOGD("%s", buffer);
        exit(1);
    }
    LOGD("error 323 check");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);
    checkCL(err);

    // Create the input and output arrays in device memory for our calculation (혹시 에러나면 * 4)
    d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * pixelsCount, NULL, &err);
    checkCL(err);
    d_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * pixelsCount, NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0,
                                 sizeof(unsigned char) * pixelsCount, tempPixels, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src));
    checkCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst));
    checkCL(clSetKernelArg(kernel, 2, sizeof(int), &info.width));
    checkCL(clSetKernelArg(kernel, 3, sizeof(int), &info.height));

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                   0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0,
                                sizeof(unsigned char) * pixelsCount, src, 0, NULL, NULL ));

    // release OpenCL resources
    checkCL(clReleaseMemObject(d_src));
    checkCL(clReleaseMemObject(d_dst));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseKernel(kernel));
    checkCL(clReleaseCommandQueue(queue));
    checkCL(clReleaseContext(context));

    AndroidBitmap_unlockPixels(env, bitmap);

    free(tempPixels);
    return bitmap;
}