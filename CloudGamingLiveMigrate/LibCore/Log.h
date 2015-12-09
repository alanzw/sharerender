#ifndef __LOG__
#define __LOG__
#define SCREEN_LOG

#include <fstream>
#include <sstream>
#include <ostream>
using namespace std;
namespace cg{
	namespace core{
		class Log {
		public:
			static void init(const char* fname);
			static void log(const char* text, ...);
			static void slog(const char* text, ...);
			static void logscreen(const char * text, ...);
			static void close();

		private:
			static char fname_[100];
			static ofstream fs_;
			static int is_init_;
		};

	}
}

#endif