#include "App_comp_pbrtvolume.h"

#ifdef COMPGL
	#include "COMPGL_PBRTVOLUME.h"
#elif COMPCL
	#include "COMPCL_PBRTVOLUME.h"
#elif COMPCU
	#include "COMPCU_PBRTVOLUME.h"
#endif

#include "system/log.h"
#include "system/timer.h"
#include "system/filesystem.h"

static void
loadCamera(const std::string& fn, int& width, int& height, float* raster2camera, float* camera2world) {
	FILE *f = fopen(fn.c_str(), "r");
	if (!f) {
		perror(fn.c_str());
		exit(1);
	}
	if (fscanf(f, "%d %d", &width, &height) != 2) {
		fprintf(stderr, "Unexpected end of file in camera file\n");
		exit(1);
	}

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (fscanf(f, "%f", &raster2camera[4 * i + j]) != 1) {
				fprintf(stderr, "Unexpected end of file in camera file\n");
				exit(1);
			}
		}
	}
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (fscanf(f, "%f", &camera2world[4 * i + j]) != 1) {
				fprintf(stderr, "Unexpected end of file in camera file\n");
				exit(1);
			}
		}
	}
	fclose(f);
}


/* Load a volume density file.  Expects the number of x, y, and z samples
   as the first three values (as integer strings), then x*y*z
   floating-point values (also as strings) to give the densities.  */
static float *
loadVolume(const std::string& fn, int& vX, int& vY, int& vZ) {
	FILE *f = fopen(fn.c_str(), "r");
	if (!f) {
		perror(fn.c_str());
		exit(1);
	}

	if (fscanf(f, "%d %d %d", &vX, &vY, &vZ) != 3) {
		fprintf(stderr, "Couldn't find resolution at start of density file\n");
		exit(1);
	}

	int count = vX * vY * vZ;
	float *v = new float[count];
	for (int i = 0; i < count; ++i) {
		if (fscanf(f, "%f", &v[i]) != 1) {
			fprintf(stderr, "Unexpected end of file at %d'th density value\n", i);
			exit(1);
		}
	}

	return v;
}

App AppInstance;

App::App()
{
	isDynamicCamera = false;

	useInterop = false;

	useCLGDevice = true;
	useCLPId = -1;
	useCLDId = -1;

	calculator = NULL;

	currWidth = 800;
	currHeight = 800;

	texId = 0;

	{
		launchW = 0;
		launchH = 0;

		wgsX = 8;
		wgsY = 8;
		wgsZ = 1;
	}

	raster2camera.resize(16);
	camera2world.resize(16);
}

App::~App()
{

}

void App::PrintCommandLineHelp()
{
	GLUTApplication::PrintCommandLineHelp();

	LOG( "Options" );

#ifdef COMPCL
	LOG( "\t--ei useCLP = select OpenCL platform with index" );
	LOG( "\t--ei useCLD = select OpenCL device with index" );
	LOG( "\t--ez useCLGL = select OpenCL device for current OpenGL context" );
	LOG( "" );
#endif

	LOG("\t--ei sampleNum = set number of samples");

	LOG("\t--ei launchW = set image size dim0");
	LOG("\t--ei launchH = set image size dim1");

	LOG( "\t--ei wgsX = set work-group size dim0" );
	LOG( "\t--ei wgsY = set work-group size dim1" );
	LOG( "\t--ei wgsZ = set work-group size dim2" );

	LOG("\t--ez dynCam = enabled mouse+keyboard camera");
	
	LOG( "" );
}

void App::initScene()
{
	loadCamera(FileSystem::GetRawFolder() + "pbrtvolume/camera.dat", launchW, launchH, raster2camera.data(), camera2world.data());
	volumeData = loadVolume(FileSystem::GetRawFolder() + "pbrtvolume/density_highres.vol", vX, vY, vZ);

	currWidth = launchW;
	currHeight = launchH;
}

void App::initCamera()
{
	// final camera params
	Vec4f lookfrom = Vec4f(13.0f, 2.0f, 3.0f, 1.0f);
	Vec4f lookat = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
	Vec4f vup = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
	
	if (isDynamicCamera)
	{
		float aspect = (float)(launchW) / (float)(launchH);

		mainCamera.setViewParams(lookfrom, lookat, vup);
		mainCamera.setProjParams(vfov, aspect, 0.25f, 100.0f);
		mainCamera.updateRays();

		appCamera = &mainCamera;

		origin = mainCamera.getPosition();
		w = mainCamera.getDirection();
		u = mainCamera.getRight();
		v = mainCamera.getUp();
	}
	else
	{
		origin = lookfrom;
		w = normalize(lookfrom - lookat);
		u = normalize(cross(vup, w));
		v = cross(w, u);
	}
}

void App::updateCameraArray()
{
	if (isDynamicCamera)
	{
		origin = mainCamera.getPosition();
		w = mainCamera.getDirection();
		u = mainCamera.getRight();
		v = mainCamera.getUp();
	}

	float aspect = (float)(launchW) / (float)(launchH);
	float dist_to_focus = 10.0f;
	float aperture = 0.1f;
	lens_radius = aperture / 2.0f;

	float theta = vfov * static_cast<float>(M_PI / 180.0);
	float half_height = tan(theta / 2);
	float half_width = aspect * half_height;

	Vec4f lower_left_corner = origin - half_width * dist_to_focus * u - half_height * dist_to_focus * v - dist_to_focus * w;
	Vec4f horizontal = 2.0f * half_width * dist_to_focus * u;
	Vec4f vertical = 2.0f * half_height * dist_to_focus * v;

	hostCameraArray[0] = origin.x;
	hostCameraArray[1] = origin.y;
	hostCameraArray[2] = origin.z;
	hostCameraArray[3] = origin.w;

	hostCameraArray[4] = lower_left_corner.x;
	hostCameraArray[5] = lower_left_corner.y;
	hostCameraArray[6] = lower_left_corner.z;
	hostCameraArray[7] = lower_left_corner.w;

	hostCameraArray[8] = horizontal.x;
	hostCameraArray[9] = horizontal.y;
	hostCameraArray[10] = horizontal.z;
	hostCameraArray[11] = horizontal.w;

	hostCameraArray[12] = vertical.x;
	hostCameraArray[13] = vertical.y;
	hostCameraArray[14] = vertical.z;
	hostCameraArray[15] = vertical.w;

	hostCameraArray[16] = u.x;
	hostCameraArray[17] = u.y;
	hostCameraArray[18] = u.z;
	hostCameraArray[19] = u.w;

	hostCameraArray[20] = v.x;
	hostCameraArray[21] = v.y;
	hostCameraArray[22] = v.z;
	hostCameraArray[23] = v.w;

	hostCameraArray[24] = w.x;
	hostCameraArray[25] = w.y;
	hostCameraArray[26] = w.z;
	hostCameraArray[27] = w.w;

	hostCameraArray[28] = lens_radius;
	hostCameraArray[29] = 0.0f;
	hostCameraArray[30] = 0.0f;
	hostCameraArray[31] = 0.0f;
}

void App::Initialize()
{
	ReadCmdArg(STRING(useInterop), useInterop);

	ReadCmdArg("useCLGL", useCLGDevice);
	ReadCmdArg("useCLP", useCLPId);
	ReadCmdArg("useCLD", useCLDId);

	ReadCmdArg(STRING(launchW), launchW);
	ReadCmdArg(STRING(launchH), launchH);

	ReadCmdArg(STRING(wgsX), wgsX);
	ReadCmdArg(STRING(wgsY), wgsY);
	ReadCmdArg(STRING(wgsZ), wgsZ);

	ReadCmdArg("dynCam", isDynamicCamera);
	
	MeshFile mfFSQ;
	mfFSQ.createFSQPosTex();
	fsqMesh.create( mfFSQ );

	fsqProgram.addFile("fsq.vert", GL_VERTEX_SHADER);
	fsqProgram.addFile("fsq_texcol.frag", GL_FRAGMENT_SHADER);
	fsqProgram.buildProgram();

	{
		glGenTextures(1, &texId);
		glBindTexture(GL_TEXTURE_2D, texId);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, launchW, launchH, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	{
		initScene();
		initCamera();
		updateCameraArray();
	}

	outputFLT = new float[4*launchW*launchH];

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

#ifdef COMPGL
	api = "GL";
	calculator = new COMPGL_PBRTVOLUME;
#elif COMPCL
	api = "CL";
	calculator = new COMPCL_PBRTVOLUME;
#elif COMPCU
	api = "CU";
	calculator = new COMPCU_PBRTVOLUME;
#endif

	calculator->cameraArray = hostCameraArray;
	calculator->volumeData = volumeData;
	calculator->raster2camera = raster2camera.data();
	calculator->camera2world = camera2world.data();
	calculator->vX = vX;
	calculator->vY = vY;
	calculator->vZ = vZ;

	calculator->outputFLT = outputFLT;

	calculator->isDynamicCamera = isDynamicCamera;

	calculator->useInterop = useInterop;
	calculator->interopId = texId;

	calculator->launchW = launchW;
	calculator->launchH = launchH;
	
	calculator->wgsX = wgsX;
	calculator->wgsY = wgsY;
	calculator->wgsZ = wgsZ;

	calculator->useCLGDevice = useCLGDevice;
	calculator->useCLPId = useCLPId;
	calculator->useCLDId = useCLDId;

	calculator->init();

	// refresh interop status based on device override
	useInterop = calculator->useInterop;

	LOG(LogLine() << "GLOBAL_WORK_SIZE: " << calculator->launchW << " x " << calculator->launchH << "\n");
	LOG(LogLine() << "LOCAL_WORK_SIZE: " << calculator->wgsX << " x " << calculator->wgsY << "\n");
	LOG(LogLine() << "INTEROP: " << (useInterop ? "ENABLED" : "DISABLED") << "\n");
}

void App::Update()
{
	if (isDynamicCamera)
		updateCameraArray();
}

void App::Render()
{
	{
		// compute
		{
			Timer timer;
			timer.start();

			calculator->compute();

			timer.stop();
			double dt = timer.getElapsedTimeInMilliSec();

			//std::cout << "COMPUTE UPDATE " << api << ": " << formatDouble(dt, 8, 4, ' ') << " ms" << std::endl;
		}

		// download
		if(useInterop == false)
		{
			Timer timer;
			timer.start();

			calculator->download();

			timer.stop();
			double dt = timer.getElapsedTimeInMilliSec();

			//std::cout << "COMPUTE DOWNLOAD" << ": " << dt << " ms" << std::endl;
		}

		// upload
		if(useInterop == false)
		{
			Timer timer;
			timer.start();

			glBindTexture(GL_TEXTURE_2D, calculator->interopId);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, launchW, launchH, 0, GL_RGBA, GL_FLOAT, outputFLT);
			glBindTexture(GL_TEXTURE_2D, 0);

			timer.stop();
			double dt = timer.getElapsedTimeInMilliSec();

			//std::cout << "RENDER UPLOAD" << ": " << dt << " ms" << std::endl;
		}

		//std::cout << std::endl;
	}

	float renderRatio = (float)launchW / (float)launchH;
	float windowRatio = (float)currWidth / (float)currHeight;

	GLint vpX, vpY;
	GLint vpW, vpH;

	if(windowRatio >= renderRatio)
	{
		vpX = static_cast<GLint>((currWidth - currHeight * renderRatio) / 2);
		vpY = 0;

		vpW = static_cast<GLsizei>(currHeight * renderRatio);
		vpH = currHeight;
	}
	else
	{
		vpX = 0;
		vpY = static_cast<GLint>((currHeight - currWidth / renderRatio) / 2);

		vpW = currWidth;
		vpH = static_cast<GLsizei>(currWidth / renderRatio);
	}

	glViewport( vpX, vpY, vpW, vpH );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	{
		fsqProgram.useProgram();

		fsqProgram.setIntValue(0, "imgTex");
		glActiveTexture(GL_TEXTURE0 + 0);
		
		glBindTexture(GL_TEXTURE_2D, texId);
		
		fsqMesh.render( fsqProgram.getAttribLocations() );

		glBindTexture(GL_TEXTURE_2D, 0);

		glUseProgram(0);
	}

	CHECK_GL;
}

void App::Terminate()
{
	DEALLOC(calculator);

	DEALLOC_ARR(outputFLT);

	fsqMesh.clear();
	fsqProgram.clear();

	if(texId)
	{
		glDeleteTextures(1, &texId);
		texId = 0;
	}

	CHECK_GL;
}

std::string App::GetName()
{
#ifdef COMPGL
	return std::string("comp_pbrtvolume_gl");
#elif COMPCL
	return std::string("comp_pbrtvolume_cl");
#elif COMPCU
	return std::string("comp_pbrtvolume_cu");
#endif
}
