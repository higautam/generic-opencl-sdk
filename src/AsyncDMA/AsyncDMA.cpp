#include "AsyncDMA.hpp"

const size_t blockX = 256;
const size_t blockY = 256;
const size_t blockZ = 512;
const size_t chunk = 16;
const size_t size_S = blockX * blockY * blockZ * sizeof(cl_float4); //256MB
const size_t size_s = blockX * blockY * chunk * sizeof(cl_float4);
static const int WindowWidth = 80;

const size_t MaxQueues = 3;

static const char* strKernel =
"__kernel void dummy(__global float4* out)  \n"
"{                                          \n"
"   uint id = get_global_id(0);             \n"
"   float4 value = (float4)(1.0f, 2.0f, 3.0f, 4.0f);  \n"
"   uint factorial = 1;                     \n"
"   for (uint i = 1; i < (id / 0xc00); ++i)\n"
"   {                                       \n"
"       factorial *= i;                     \n"
"   }                                       \n"
"   out[id] = value * factorial;            \n"
"}                                          \n";

#define CHECK_RESULT(test, msg)\
	if(test)\
	{\
		std::cerr << __FILE__ << __LINE__\
			<< "  Message: " << msg\
			<< "  Error Code: " << test;\
		exit(1);\
	}

class ProfileQueue {
public:
    enum Operation
    {
        Write = 0,
        Execute,
        Read,
        Total
    };

    static const char* OperationName[Total];
    static const char StartCommand[Total];
    static const char ExecCommand[Total]; 

    ProfileQueue() {}
    ~ProfileQueue()
    {
        for (size_t op = 0; op < Total; ++op) {
            for (size_t idx = 0; idx < events_[op].size(); ++idx) {
                clReleaseEvent(events_[op][idx]);
            }
        }
    }

    void addEvent(Operation op, cl_event event)
    {
        events_[op].push_back(event);
    }

    //To find the time taken by the most time taking command queue
    void findMinMax(cl_long* min_time, cl_long* max_time)
    {
        // Find time min/max ranges for the frame scaling
        for (size_t op = 0; (op < ProfileQueue::Total); ++op) 
		{
            cl_long time;
            if (events_[op].size() == 0) continue;
            clGetEventProfilingInfo(events_[op][0], CL_PROFILING_COMMAND_START,
                sizeof(cl_long), &time, NULL);
            if (0 == *min_time) {
                *min_time = time;
            }
            else {
                *min_time = std::min<cl_long>(*min_time, time);
            }
            clGetEventProfilingInfo(events_[op][events_[op].size() - 1],
                CL_PROFILING_COMMAND_END, sizeof(cl_long), &time, NULL);
            if (0 == *max_time) {
                *max_time = time;
            }
            else {
                *max_time = std::max<cl_long>(*max_time, time);
            }
        }
    }

    void display(cl_long start, cl_long finish)
    {
        std::string graph;
        graph.resize(WindowWidth + 1);
        graph[WindowWidth] = '\x0';
        cl_long timeFrame = finish - start;
        cl_long interval = timeFrame / WindowWidth;

        // Find time min/max ranges for the frame scaling
        for (size_t op = 0; (op < Total); ++op) {
            if (events_[op].size() == 0) continue;
            cl_long timeStart, timeEnd;
            int begin = 0, end = 0;
            for (size_t idx = 0; idx < events_[op].size(); ++idx) {
                bool cutStart = false;
                clGetEventProfilingInfo(events_[op][idx], CL_PROFILING_COMMAND_START,
                    sizeof(cl_long), &timeStart, NULL);
                clGetEventProfilingInfo(events_[op][idx], CL_PROFILING_COMMAND_END,
                    sizeof(cl_long), &timeEnd, NULL);

                // Continue if out of the frame scope
                if (timeStart >= finish) continue;
                if (timeEnd <= start) continue;

                if (timeStart <= start) {
                    timeStart = start;
                    cutStart = true;
                }

                if (timeEnd >= finish) {
                    timeEnd = finish;
                }

                // Readjust time to the frame
                timeStart -= start;
                timeEnd -= start;
                timeStart = static_cast<cl_long>(
                    floor(static_cast<float>(timeStart) / interval + 0.5f));
                timeEnd = static_cast<cl_long>(
                    floor(static_cast<float>(timeEnd) / interval + 0.5f));
                begin = static_cast<int>(timeStart);
                // Idle from end to begin
                for (int c = end; c < begin; ++c) {
                    graph[c] = '-';
                }
                end = static_cast<int>(timeEnd);
                for (int c = begin; c < end; ++c) {
                    if ((c == begin) && !cutStart) {
                        graph[c] = StartCommand[op];
                    }
                    else {
                        graph[c] = ExecCommand[op];
                    }
                }
                if ((begin == end) && (end < WindowWidth)) {
                    graph[begin] = '+';
                }
            }
            if (end < WindowWidth) {
                for (int c = end; c < WindowWidth; ++c) {
                    graph[c] = '-';
                }
            }
            printf("%s\n", graph.c_str());
        }
    }

private:
    // Profiling events
    std::vector<cl_event>   events_[Total];
};

const char* ProfileQueue::OperationName[Total] = { "BufferWrite", "KernelExecution", "BufferRead" };
const char ProfileQueue::StartCommand[Total] = { 'W', 'X', 'R' };
const char ProfileQueue::ExecCommand[Total] = { '>', '#', '<' };


class Profile {
public:
    Profile(bool profEna, int numQueues)
        : profileEna_(profEna)
        , numQueues_(numQueues)
        , min_(0)
        , max_(0)
        , execTime_(0) {}

    ~Profile() {}

    void addEvent(int queue, ProfileQueue::Operation op, cl_event event)
    {
        if (profileEna_) {
            profQueue[queue].addEvent(op, event);
        }
    }

    cl_long findExecTime()
    {
        if (execTime_ != 0) return execTime_;
        for (int q = 0; q < numQueues_; ++q) {
            profQueue[q].findMinMax(&min_, &max_);
        }
        execTime_ = max_ - min_;
        return execTime_;
    }

    void display(cl_long start, cl_long finish)
    {
        if (!profileEna_) return;
        printf("\n ----------- Time frame %.3f (us), scale 1:%.0f\n",
            (float)(finish - start) / 1000, (float)(finish - start) / (1000 * WindowWidth));
        for (size_t op = 0; (op < ProfileQueue::Total); ++op) {
            printf("%s - %c%c; ", ProfileQueue::OperationName[op],
                ProfileQueue::StartCommand[op], ProfileQueue::ExecCommand[op]);
        }
        printf("\n");
        for (int q = 0; q < numQueues_; ++q) {
            printf("CommandQueue #%d\n", q);
            profQueue[q].display(min_ + start, min_ + finish);
        }
    }

private:
    bool    profileEna_;
    int     numQueues_;     //!< Total number of queues
    cl_long min_;           //!< Min HW timestamp
    cl_long max_;           //!< Max HW timestamp
    cl_long execTime_;      //!< Profile time
    ProfileQueue    profQueue[MaxQueues];
};

OCLPerfDoubleDMA::OCLPerfDoubleDMA(unsigned int test, bool isProfilingEnabled, bool useUHP) 
	: test_(test),
	isProfilingEnabled_(isProfilingEnabled),
	useUHP_(useUHP),
	hostPtr_(NULL)
{
    failed_ = false;
}

cl_uint OCLPerfDoubleDMA::NumSubTests()
{
    return (2 * MaxQueues); // Iterate over 1/2/3 queue, with/without using kernel
}

OCLPerfDoubleDMA::~OCLPerfDoubleDMA()
{
}

void OCLPerfDoubleDMA::open(cl_uint platformIdx, cl_uint deviceId)
{
    deviceId_ = deviceId;

    cl_uint numPlatforms = 0;
    error_ = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_RESULT((error_), "clGetPlatformIDs failed");
    CHECK_RESULT((numPlatforms == 0), "No platform found");

    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_, "clGetPlatformIDs failed");

    cl_platform_id platform = 0;
    platform = platforms[platformIdx];

    delete [] platforms;

    CHECK_RESULT((platform == 0), "AMD Platform not found");

    error_ = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount_);
    CHECK_RESULT(error_, "clGetDeviceIDs() failed");
    
    devices_ = new cl_device_id[deviceCount_];
    error_ = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount_, devices_, NULL);
    CHECK_RESULT(error_, "clGetDeviceIDs() failed");

    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties) platform, 0};
    context_ = clCreateContext( props, deviceCount_, devices_, NULL, 0, &error_);
    CHECK_RESULT((error_), "clCreateContext failed");

    platform_ = platform;

    cl_device_type deviceType;
    error_ = clGetDeviceInfo(devices_[deviceId],
        CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    CHECK_RESULT((error_), "CL_DEVICE_TYPE failed");

    if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
        printf("GPU device is required for this test!\n");
        failed_ = true;
        return;        
    }
    program_ = clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
    CHECK_RESULT((error_), "clCreateProgramWithSource()  failed");
    error_ = clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
    if (error_ != CL_SUCCESS) {
        char programLog[1024];
        clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024, programLog, 0);
        printf("\n%s\n",programLog);
        fflush(stdout);
    }
    CHECK_RESULT((error_), "clBuildProgram() failed");
    kernel_ = clCreateKernel(program_, "dummy", &error_);
    CHECK_RESULT((error_), "clCreateKernel() failed");

    size_t  bufSize = size_s;
    cl_mem  buffer;
    
	size_t  numBufs = (test_ % MaxQueues) + 1;
    for (size_t b = 0; b < numBufs; ++b) {
        buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            bufSize, NULL, &error_);
        CHECK_RESULT((error_), "clCreateBuffer() failed");
        buffers_.push_back(buffer);
    }

	cl_mem_flags flags = CL_MEM_READ_WRITE;

	if(useUHP_)
	{
		hostPtr_ = new char[size_S];
		flags = flags | CL_MEM_USE_HOST_PTR;

		buffer = clCreateBuffer(context_, flags, size_S, hostPtr_, &error_);
		CHECK_RESULT(error_, "clCreateBuffer Failed");
	}
	else
	{
		flags = flags | CL_MEM_ALLOC_HOST_PTR;
		
		buffer = clCreateBuffer(context_, flags,
			size_S, NULL, &error_);
		CHECK_RESULT((error_), "clCreateBuffer() failed");
	}
    
	buffers_.push_back(buffer);
}

void
OCLPerfDoubleDMA::run()
{
    if (failed_) {
        return;
    }
    CPerfCounter timer;
    const int   numQueues = (test_ % MaxQueues) + 1;
    const bool  useKernel = ((test_ / MaxQueues) > 0);
    const int   numBufs = numQueues;
    Profile     profile(isProfilingEnabled_, numQueues);

    std::vector<cl_command_queue> cmdQueues(numQueues);
    int q;
    cl_command_queue_properties qProp = (isProfilingEnabled_) ? CL_QUEUE_PROFILING_ENABLE : 0;
    for (q = 0; q < numQueues; ++q) {
        cl_command_queue cmdQueue = clCreateCommandQueue(
            context_, devices_[deviceId_], qProp, &error_);
        CHECK_RESULT((error_), "clCreateCommandQueue() failed");
        cmdQueues[q] = cmdQueue;
    }
    
    float *Data_s = (float*)clEnqueueMapBuffer(cmdQueues[0],
        buffers_[numBufs], CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, size_S, 0, NULL, NULL, &error_);
    CHECK_RESULT((error_), "clEnqueueMapBuffer failed");
	memset(Data_s, 1, size_S);
    size_t  gws[1] = { size_s / (4 * sizeof(float)) };
    size_t  lws[1] = { 256 };

    // Warm-up
    for (q = 0; q < numQueues; ++q) {
        error_ |= clEnqueueWriteBuffer(cmdQueues[q],
            buffers_[q], CL_FALSE, 0, size_s, (char*)Data_s, 0, NULL, NULL);
        error_ |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*) &buffers_[q]);
        error_ |= clEnqueueNDRangeKernel(cmdQueues[q],
            kernel_, 1, NULL, gws, lws, 0, NULL, NULL);
        error_ |= clEnqueueReadBuffer(cmdQueues[q],
            buffers_[q], CL_FALSE, 0, size_s, (char*)Data_s, 0, NULL, NULL);
        error_ |= clFinish(cmdQueues[q]);
    }

    size_t s_done = 0;
    cl_event r[MaxQueues] = {0}, w[MaxQueues] = {0}, x[MaxQueues] = {0};

    /*----------  pass2:  copy Data_s to and from GPU Buffers ----------*/
    s_done = 0;
    timer.Reset();
    timer.Start();
    int idx = numBufs - 1;
    // Start from the last so read/write won't go to the same DMA when kernel is executed
    q = numQueues - 1;
    size_t iter = 0;
    while( 1 )  {
        if (0 == r[idx]) {
            error_ |= clEnqueueWriteBuffer(cmdQueues[q],
                buffers_[idx], CL_FALSE, 0, size_s, (char*)Data_s+s_done, 0, NULL, &w[idx]);
        }
        else {
            error_ |= clEnqueueWriteBuffer(cmdQueues[q],
                buffers_[idx], CL_FALSE, 0, size_s, (char*)Data_s+s_done, 1, &r[idx], &w[idx]);
            if (!isProfilingEnabled_) { 
                error_ |= clReleaseEvent(r[idx]);
            }
        }
        profile.addEvent(q, ProfileQueue::Write, w[idx]);

        if (useKernel) {
            // Change the queue
            ++q %= numQueues;
            // Implicit flush of DMA engine on kernel start, because memory dependency
            error_ |= clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*) &buffers_[idx]);
            error_ |= clEnqueueNDRangeKernel(cmdQueues[q],
                kernel_, 1, NULL, gws, lws, 1, &w[idx], &x[idx]);
            if (!isProfilingEnabled_) { 
                error_ |= clReleaseEvent(w[idx]);
            }
            profile.addEvent(q, ProfileQueue::Execute, x[idx]);
        }

        // Change the queue
        ++q %= numQueues;
        error_ |= clEnqueueReadBuffer(cmdQueues[q],
            buffers_[idx], CL_FALSE, 0, size_s, (char*)Data_s+s_done, 1,
            (useKernel) ? &x[idx] : &w[idx], &r[idx]);
        if (!isProfilingEnabled_) { 
            error_ |= clReleaseEvent((useKernel) ? x[idx] : w[idx]);
        }
        profile.addEvent(q, ProfileQueue::Read, r[idx]);

        if ((s_done += size_s) >= size_S) {
            if (!isProfilingEnabled_) { 
                error_ |= clReleaseEvent(r[idx]);
            }
            break;
        }
        ++iter;
        ++idx %= numBufs;
        ++q %= numQueues;
    }

    for (q = 0; q < numQueues; ++q) {
        error_ |= clFinish(cmdQueues[q]);
    }
    timer.Stop();

    error_ = clEnqueueUnmapMemObject(cmdQueues[0],
        buffers_[numBufs], Data_s, 0, NULL, NULL);

    error_ |= clFinish(cmdQueues[0]);
    CHECK_RESULT((error_), "Execution failed");

    cl_long gpuTimeFrame = profile.findExecTime();
    cl_long oneIter = gpuTimeFrame / iter;

    // Display 4 iterations in the middle
    cl_long startFrame = oneIter * (iter/2 - 2);
    cl_long finishFrame = oneIter * (iter/2 + 2);
    profile.display(startFrame, finishFrame);

    for (q = 0; q < numQueues; ++q) {
        error_ = clReleaseCommandQueue(cmdQueues[q]);
        CHECK_RESULT((error_), "clReleaseCommandQueue() failed");
    }

    double GBytes = (double)(2*size_S)/(double)(1024*1024*1024);

    std::stringstream stream;
    if (useKernel) {
        stream << "Write/Kernel/Read operation ";
    }
    else {
        stream << "Write/Read operation ";
    }
    stream << numQueues << " queue; profiling " <<
        ((isProfilingEnabled_) ? "enabled" : "disabled");

	stream << ((useUHP_) ? " using UHP" : " using AHP") << ": "; 
    
    stream.flags(std::ios::right | std::ios::showbase);
    std::cout << stream.str() << static_cast<float>(GBytes / timer.GetElapsedTime()) << " GB/s\n";
}

void
OCLPerfDoubleDMA::close()
{
    for (unsigned int i = 0; i < buffers().size(); ++i) {
        error_ = clReleaseMemObject(buffers()[i]);
    }
    buffers_.clear();

    if (kernel_ != 0) {
        error_ = clReleaseKernel(kernel_);
    }

    if (program_ != 0) {
        error_ = clReleaseProgram(program_);
    }

    if (context_) {
        error_ = clReleaseContext(context_);
    }

    if (devices_) {
        delete [] devices_;
    }

	if(hostPtr_) {
		delete hostPtr_;
	}
}


int 
main(int argc, char * argv[])
{
	bool use_UHP, profiling_enabled;
    // NumSubTests return 3 x 2 = 6 tests. Using 1/2/3 command queues (3) and Doing asyncDMA with/witout a kernel execution (2)
	for (cl_uint i = 0; i < OCLPerfDoubleDMA::NumSubTests(); ++i) 
	{
		// Running application using ALLOC_HOST_PTR flag, and profiling enabled
		use_UHP = false, profiling_enabled = false;
	    OCLPerfDoubleDMA* test0 = new OCLPerfDoubleDMA(i, profiling_enabled, use_UHP);
        test0->open(0, 0);
        test0->run();
        test0->close();
        delete test0;

		// Running application using ALLOC_HOST_PTR flag, and profiling disabled
		use_UHP = false, profiling_enabled = true;
		OCLPerfDoubleDMA* test1 = new OCLPerfDoubleDMA(i, profiling_enabled, use_UHP);
        test1->open(0, 0);
        test1->run();
        test1->close();
        delete test1;

		// Running application using USE_HOST_PTR flag, and profiling disabled
		use_UHP = true, profiling_enabled = false;
		OCLPerfDoubleDMA* test2 = new OCLPerfDoubleDMA(i, profiling_enabled, use_UHP);
        test2->open(0, 0);
        test2->run();
        test2->close();
        delete test2;

		// Running application using USE_HOST_PTR flag, and profiling enabled
		use_UHP = true, profiling_enabled = true;
		OCLPerfDoubleDMA* test3 = new OCLPerfDoubleDMA(i, profiling_enabled, use_UHP);
        test3->open(0, 0);
        test3->run();
        test3->close();
        delete test3;
    }
    return 0;
}
