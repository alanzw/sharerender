#include <iostream>
#include <vector>
#include <string>

using namespace std;

namespace cg{
	namespace core{
		string trim(const string & str);
		int split(const string& str, vector<string>& ret_, string sep = ",");
		string replace(const string& str, const string& src, const string& dest);

	}
}