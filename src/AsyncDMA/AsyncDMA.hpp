#ifndef _ASYNC_DMA_H_
#define _ASYNC_DMA_H_

#include <CL/cl.h>
#include "CL/cl_ext.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <cstring>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <time.h>

#if defined (_WIN32) || defined(_WIN64)
#include <windows.h>
typedef __int64 i64 ;
#else
#include <sys/time.h>
#include <time.h>
#endif


class CPerfCounter {

public:
    CPerfCounter();
    ~CPerfCounter();
    void Start(void);
    void Stop(void);
    void Reset(void);
    double GetElapsedTime(void);

private:
#if defined(_WIN32) || (_WIN64)
    i64 _freq, _clocks, _start;
#else
	struct timespec _start, _res, _end;
	double _time;
#endif
};


#if defined(_WIN32) || defined(_WIN64)
CPerfCounter::CPerfCounter() : _clocks(0), _start(0)
{
    QueryPerformanceFrequency((LARGE_INTEGER *)&_freq);
}

CPerfCounter::~CPerfCounter()
{
    // EMPTY!
}

void
CPerfCounter::Start(void)
{
    QueryPerformanceCounter((LARGE_INTEGER *)&_start);
}

void
CPerfCounter::Stop(void)
{
    i64 n;
    QueryPerformanceCounter((LARGE_INTEGER *)&n);
    n -= _start;
    _start = 0;
    _clocks += n;
}

void
CPerfCounter::Reset(void)
{
    _clocks = 0;
}

double
CPerfCounter::GetElapsedTime(void)
{
    return (double)_clocks / (double)_freq;
}

#else //For Linux
CPerfCounter::CPerfCounter(): _time(0) 
{
    clock_getres(CLOCK_REALTIME, &_res);
}

CPerfCounter::~CPerfCounter()
{
    // EMPTY!
}

void
CPerfCounter::Start(void)
{
    clock_gettime(CLOCK_REALTIME, &_start);
}

void
CPerfCounter::Stop(void)
{
	clock_gettime(CLOCK_REALTIME, &_end);

	if (_end.tv_nsec > _start.tv_nsec)
    	{
        	_time = (double)(_end.tv_nsec - _start.tv_nsec)/(double)1000000000;
       		_time += (_end.tv_sec - _start.tv_sec);
    	} else 
	{
        	_time = (double)1 +
                (double)(_end.tv_nsec - _start.tv_nsec)/(double)1000000000;
        	_time += ((_end.tv_sec-1) - _start.tv_sec);
    	}

	_start.tv_nsec = 0, _start.tv_sec = 0;
}

void
CPerfCounter::Reset(void)
{
_time = 0;
}

double
CPerfCounter::GetElapsedTime(void)
{
    return _time;
}
#endif


class OCLPerfDoubleDMA
{
public:
    static cl_uint NumSubTests();

    OCLPerfDoubleDMA(cl_uint test, bool isProfilingEnabled, bool useUHP);
    ~OCLPerfDoubleDMA();

public:
    void    open(cl_uint platformIdx, cl_uint deviceID);
    void    run();
    void    close();

private:
    bool    failed_;
    cl_uint test_;
	bool isProfilingEnabled_;
	bool useUHP_;

    const std::vector<cl_mem>& buffers() const { return buffers_; }

    // Common data of any CL program
    cl_int                      error_;
    cl_uint                     deviceCount_;
    cl_device_id*               devices_;
    cl_platform_id              platform_;
    cl_context                  context_;
    cl_uint                     deviceId_;

    cl_program                  program_;
    cl_kernel                   kernel_;
    std::vector<cl_mem>         buffers_;
	char* hostPtr_;

};

#endif  /* #ifndef _ASYNC_DMA_ */
