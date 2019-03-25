#include "clutil/clbase.h"

int main(int argc, char** argv)
{
	(void)argc;
	(void)argv;

	int clewOK = initClew();
	if (clewOK != 0)
	{
		std::cout << "initClew() failed!" << std::endl;
		exit(-1);
	}

	OpenCLUtil cl;
	cl.init();
	cl.print();

	return 0;
}