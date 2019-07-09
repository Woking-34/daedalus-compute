#include "App_comp_rtow.h"

#include "COMP_RTOW_TYPES.h"

#ifdef COMPGL
	#include "COMPGL_RTOW.h"
#elif COMPCL
	#include "COMPCL_RTOW.h"
#elif COMPCU
	#include "COMPCU_RTOW.h"
#endif

#include "system/log.h"
#include "system/timer.h"
#include "system/filesystem.h"

// https://gist.github.com/mortennobel/8665258
#ifdef _WIN32
#include "drand48.h"
#endif

using namespace daedalus;

App AppInstance;

App::App()
{
	isDynamicCamera = false;

	useInterop = true;

	useCLGDevice = true;
	useCLPId = -1;
	useCLDId = -1;

	calculator = NULL;

	currWidth = 800;
	currHeight = 800;

	texId = 0;

	{
		sampleNum = 2;

		launchW = 1200;
		launchH = 800;

		wgsX = 8;
		wgsY = 8;
		wgsZ = 1;
	}
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
	int currMatId = 0;

	{
		//list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));

		rtow_sphere currSphere;
		currSphere.posX = 0.0f;
		currSphere.posY = -1000.0f;
		currSphere.posZ = 0.0f;
		currSphere.rad = 1000.0f;
		currSphere.mat_ptr = currMatId++;

		sphereArrayVecHost.push_back(currSphere);

		rtow_material currMaterial;
		currMaterial.albedoR = 0.5f;
		currMaterial.albedoG = 0.5f;
		currMaterial.albedoB = 0.5f;
		currMaterial.fuzz = 0.0f;
		currMaterial.matType = 0;

		materialArraVecHost.push_back(currMaterial);
	}

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = static_cast<float>(drand48());
			Vec4f center(a + 0.9f*static_cast<float>(drand48()), 0.2f, b + 0.9f*static_cast<float>(drand48()), 1.0f);
			if (length(center - Vec4f(4.0f, 0.2f, 0.0f, 1.0f)) > 0.9f)
			{
				if (choose_mat < 0.8f)
				{
					// diffuse
					//list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));

					rtow_sphere currSphere;
					currSphere.posX = center.x;
					currSphere.posY = center.y;
					currSphere.posZ = center.z;
					currSphere.rad = 0.2f;
					currSphere.mat_ptr = currMatId++;

					sphereArrayVecHost.push_back(currSphere);

					float r0 = static_cast<float>(drand48());
					float r1 = static_cast<float>(drand48());
					float r2 = static_cast<float>(drand48());
					float r3 = static_cast<float>(drand48());
					float r4 = static_cast<float>(drand48());
					float r5 = static_cast<float>(drand48());

					rtow_material currMaterial;
					currMaterial.albedoR = r0*r1;
					currMaterial.albedoG = r2*r3;
					currMaterial.albedoB = r4*r5;
					currMaterial.fuzz = 0.0f;
					currMaterial.matType = 0;

					materialArraVecHost.push_back(currMaterial);
				}
				else if (choose_mat < 0.95f)
				{
					// metal
					//list[i++] = new sphere(center, 0.2, new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), 0.5*drand48()));

					rtow_sphere currSphere;
					currSphere.posX = center.x;
					currSphere.posY = center.y;
					currSphere.posZ = center.z;
					currSphere.rad = 0.2f;
					currSphere.mat_ptr = currMatId++;

					sphereArrayVecHost.push_back(currSphere);

					float r0 = static_cast<float>(drand48());
					float r1 = static_cast<float>(drand48());
					float r2 = static_cast<float>(drand48());
					float r3 = static_cast<float>(drand48());

					rtow_material currMaterial;
					currMaterial.albedoR = 0.5f*(1.0f + r0);
					currMaterial.albedoG = 0.5f*(1.0f + r1);
					currMaterial.albedoB = 0.5f*(1.0f + r2);
					currMaterial.fuzz = 0.5f*r3;
					currMaterial.matType = 1;

					materialArraVecHost.push_back(currMaterial);
				}
				else
				{
					// glass
					//list[i++] = new sphere(center, 0.2, new dielectric(1.5));

					rtow_sphere currSphere;
					currSphere.posX = center.x;
					currSphere.posY = center.y;
					currSphere.posZ = center.z;
					currSphere.rad = 0.2f;
					currSphere.mat_ptr = currMatId++;

					sphereArrayVecHost.push_back(currSphere);

					rtow_material currMaterial;
					currMaterial.albedoR = 0.0f;
					currMaterial.albedoG = 0.0f;
					currMaterial.albedoB = 0.0f;
					currMaterial.fuzz = 1.5f;
					currMaterial.matType = 2;

					materialArraVecHost.push_back(currMaterial);
				}
			}
		}
	}

	{
		//list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));

		rtow_sphere currSphere;
		currSphere.posX = 0.0f;
		currSphere.posY = 1.0f;
		currSphere.posZ = 0.0f;
		currSphere.rad = 1.0f;
		currSphere.mat_ptr = currMatId++;

		sphereArrayVecHost.push_back(currSphere);

		rtow_material currMaterial;
		currMaterial.albedoR = 0.0f;
		currMaterial.albedoG = 0.0f;
		currMaterial.albedoB = 0.0f;
		currMaterial.fuzz = 1.5f;
		currMaterial.matType = 2;

		materialArraVecHost.push_back(currMaterial);
	}

	{
		//list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));

		rtow_sphere currSphere;
		currSphere.posX = -4.0f;
		currSphere.posY = 1.0f;
		currSphere.posZ = 0.0f;
		currSphere.rad = 1.0f;
		currSphere.mat_ptr = currMatId++;

		sphereArrayVecHost.push_back(currSphere);

		rtow_material currMaterial;
		currMaterial.albedoR = 0.4f;
		currMaterial.albedoG = 0.2f;
		currMaterial.albedoB = 0.1f;
		currMaterial.fuzz = 0.0f;
		currMaterial.matType = 0;

		materialArraVecHost.push_back(currMaterial);
	}

	{
		//list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

		rtow_sphere currSphere;
		currSphere.posX = 4.0f;
		currSphere.posY = 1.0f;
		currSphere.posZ = 0.0f;
		currSphere.rad = 1.0f;
		currSphere.mat_ptr = currMatId++;

		sphereArrayVecHost.push_back(currSphere);

		rtow_material currMaterial;
		currMaterial.albedoR = 0.7f;
		currMaterial.albedoG = 0.6f;
		currMaterial.albedoB = 0.5f;
		currMaterial.fuzz = 0.0f;
		currMaterial.matType = 1;

		materialArraVecHost.push_back(currMaterial);
	}
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

	ReadCmdArg(STRING(sampleNum), sampleNum);
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

	{
		seed0 = new unsigned int[launchW*launchH];
		seed1 = new unsigned int[launchW*launchH];

		for (int i = 0; i < launchW*launchH; ++i)
		{
			seed0[i] = rand() % RAND_MAX + 1;
			seed1[i] = rand() % RAND_MAX + 1;
		}
	}

	outputFLT = new float[4*launchW*launchH];

	// init gl state
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

#ifdef COMPGL
	api = "GL";
	calculator = new COMPGL_RTOW;
#elif COMPCL
	api = "CL";
	calculator = new COMPCL_RTOW;
#elif COMPCU
	api = "CU";
	calculator = new COMPCU_RTOW;
#endif

	calculator->cameraArray = hostCameraArray;
	calculator->seed0 = seed0;
	calculator->seed1 = seed1;

	calculator->sphereNum = static_cast<int>(sphereArrayVecHost.size());
	calculator->sphereArrayHost = sphereArrayVecHost.data();
	calculator->materialNum = static_cast<int>(materialArraVecHost.size());
	calculator->materialArrayHost = materialArraVecHost.data();

	calculator->outputFLT = outputFLT;

	calculator->isDynamicCamera = isDynamicCamera;

	calculator->useInterop = useInterop;
	calculator->interopId = texId;

	calculator->sampleNum = sampleNum;

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

	DEALLOC_ARR(seed0);
	DEALLOC_ARR(seed1);

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
	return std::string("comp_rtow_gl");
#elif COMPCL
	return std::string("comp_rtow_cl");
#elif COMPCU
	return std::string("comp_rtow_cu");
#endif
}
