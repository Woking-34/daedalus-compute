#ifndef LOG_H
#define LOG_H

#include "system/platform.h"

#define LOG(Message) Log::Get().Write( Message );

#define LOG_OK(Message) Log::Get().WriteOK( Message );
#define LOG_ERR(Message) Log::Get().WriteERR( Message );
#define LOG_WARN(Message) Log::Get().WriteWARN( Message );

#define LOG_BOOL(Bool, Message)		\
	if(Bool) {						\
	LOG_OK(Message)					\
	} else {						\
	LOG_ERR(Message)				\
	}

#define LOG_INFO Log::Get().WriteInfo( __FILE__, __LINE__ );

class LogLine
{
public:
    LogLine() {}
    ~LogLine() {}

	operator std::string()
	{
		return ss.str();
	};

    template <class T>
    LogLine& operator<<(const T& thing)
	{
		ss << thing;
		return *this;
	}

private:
    std::stringstream ss;
};

class Log 
{
protected:
	Log( std::ostream& os = std::cout ): os(os) {}
	~Log() {}

	std::ostream& os;

public:
	static Log& Get();

	void Write( const char* message );
	void Write( const std::string& message );

	void WriteOK( const char* message );
	void WriteOK( const std::string& message );

	void WriteERR( const char* message );
	void WriteERR( const std::string& message );

	void WriteWARN( const char* message );
	void WriteWARN( const std::string& message );

	void WriteInfo( char const* fileName, int fileLine );
};

#endif