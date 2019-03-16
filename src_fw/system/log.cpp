#include "log.h"

Log& Log::Get()
{
	static Log log;
	return log;
}

void Log::Write( const char* message )
{
	os << message << std::endl;
	os.flush();
}
void Log::Write( const std::string& message )
{
	Write( message.c_str() );
}

void Log::WriteOK( const char* message )
{
#if defined _WIN32
	static HANDLE hstdout = GetStdHandle( STD_OUTPUT_HANDLE );
	SetConsoleTextAttribute( hstdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
#endif

	os << "[OK] ";

#if defined _WIN32
	SetConsoleTextAttribute( hstdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE );
#endif

	os << message << std::endl;
	os.flush();
}
void Log::WriteOK( const std::string& message )
{
	WriteOK( message.c_str() );
}

void Log::WriteERR( const char* message )
{
#if defined _WIN32
	static HANDLE hstdout = GetStdHandle( STD_OUTPUT_HANDLE );
	SetConsoleTextAttribute( hstdout, FOREGROUND_RED | FOREGROUND_INTENSITY );
#endif

	os << "[ERR] ";

#if defined _WIN32
	SetConsoleTextAttribute( hstdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE );
#endif

	os << message << std::endl;
	os.flush();
}
void Log::WriteERR( const std::string& message )
{
	WriteERR( message.c_str() );
}

void Log::WriteWARN( const char* message )
{
#if defined _WIN32
	static HANDLE hstdout = GetStdHandle( STD_OUTPUT_HANDLE );
	SetConsoleTextAttribute( hstdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY );
#endif

	os << "[WARN] ";

#if defined _WIN32
	SetConsoleTextAttribute( hstdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE );
#endif

	os << message << std::endl;
	os.flush();
}
void Log::WriteWARN( const std::string& message )
{
	WriteWARN( message.c_str() );
}

void Log::WriteInfo( char const* fileName, int fileLine )
{
	os << "File: " << fileName << std::endl;
	os << "Line: " << fileLine << std::endl;
	os.flush();
}