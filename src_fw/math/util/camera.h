#ifndef CAMERA_H
#define CAMERA_H

#include "math/mathbase.h"
#include "math/core/vec.h"
#include "math/core/mat.h"

#include "math/util/ray.h"

class Camera
{
public:
	Camera() {}
	~Camera() {}

	void updateRotation(float yaw, float pitch, float roll)
	{
		Vec4f up = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
		worldMat = worldMat * createTranslationMatrix(-worldMat[12], -worldMat[13], -worldMat[14]) * createRotationMatrix(up, yaw) * createTranslationMatrix(worldMat[12], worldMat[13], worldMat[14]);

		Vec4f right = getRight();
		worldMat = worldMat * createTranslationMatrix(-worldMat[12], -worldMat[13], -worldMat[14]) * createRotationMatrix(right, pitch) * createTranslationMatrix(worldMat[12], worldMat[13], worldMat[14]);

		viewMat = getMatrixtInverse(worldMat);
	}

	void updateTranslationForwardBackward(float scale)
	{
		Vec4f delta = scale * getDirection();

		worldMat[12] += delta.x;
		worldMat[13] += delta.y;
		worldMat[14] += delta.z;
		worldMat[15] += delta.w;

		viewMat = getMatrixtInverse(worldMat);
	}

	void updateTranslationLeftRight(float scale)
	{
		Vec4f delta = scale * getRight();

		worldMat[12] += delta.x;
		worldMat[13] += delta.y;
		worldMat[14] += delta.z;
		worldMat[15] += delta.w;

		viewMat = getMatrixtInverse(worldMat);
	}

	void updateTranslationUpDown(float scale)
	{
		Vec4f delta = scale * getUp();

		worldMat[12] += delta.x;
		worldMat[13] += delta.y;
		worldMat[14] += delta.z;
		worldMat[15] += delta.w;

		viewMat = getMatrixtInverse(worldMat);
	}

	void setViewParams( const Vec4f& eye, const Vec4f& center, const Vec4f& up )
	{
		viewMat = createViewLookAtMatrix(eye, center, up);

		worldMat = getMatrixtInverse(viewMat);
	}

	void setProjParams( float fovy, float aspect, float _nearPlane, float _farPlane )
	{
		float tangent = tan( deg2rad(fovy/2) );
		float height = _nearPlane * tangent;
		float width = height * aspect;

		this->_leftPlane = -width;
		this->_rightPlane = width;
		this->_bottomPlane = -height;
		this->_topPlane = height;
		this->_nearPlane = _nearPlane;
		this->_farPlane = _farPlane;

		projMat = createProjectionPerspectiveMatrix(fovy, aspect, _nearPlane, _farPlane);
	}
	void setProjParams(float _left, float _right, float _bottom, float _top, float _near, float _far)
	{
		this->_leftPlane = _left;
		this->_rightPlane = _right;
		this->_bottomPlane = _bottom;
		this->_topPlane = _top;
		this->_nearPlane = _near;
		this->_farPlane = _far;

		projMat = createProjectionPerspectiveMatrix(_left, _right, _bottom, _top, _near, _far);
	}

	void setupRayFrame(Vec4f _origin, Vec4f _target, Vec4f _up, float fov, float _near, float _far, int width, int height)
	{
		Vec4f cameraForward = _target - _origin;
		cameraForward = normalize(cameraForward);

		Vec4f cameraRight = cross(cameraForward, _up);
		cameraRight = normalize(cameraRight);

		Vec4f cameraUp = cross(cameraRight, cameraForward);
		cameraUp = normalize(cameraUp);
		
		float aspect = (float)height / (float)width;
		float imageExtentX = (float)tan(0.5f * deg2rad(fov));
		float imageExtentY = (float)tan(0.5f * aspect * deg2rad(fov));
		
		origin = _origin;
		right = cameraRight * ( 2.0f/(float)width * imageExtentX );
		up = cameraUp * ( -2.0f/(float)height * imageExtentY );
		view = cameraForward - cameraRight * imageExtentX + cameraUp * imageExtentY; 
	}

	void updatePlanes()
	{

	}
	void updateRays()
	{
		Vec4f camPos = getPosition();

		Mat44f viewprojInvMat = getMatrixtInverse(viewMat * projMat);

		Vec4f camTL = Vec4f(-1.0f, 1.0f,-1.0f, 1.0f) * viewprojInvMat;
		camTL = camTL / camTL.w;
		Vec4f camTR = Vec4f( 1.0f, 1.0f,-1.0f, 1.0f) * viewprojInvMat;
		camTR = camTR / camTR.w;
		Vec4f camBL = Vec4f(-1.0f,-1.0f,-1.0f, 1.0f) * viewprojInvMat;
		camBL = camBL / camBL.w;
		Vec4f camBR = Vec4f( 1.0f,-1.0f,-1.0f, 1.0f) * viewprojInvMat;
		camBR = camBR / camBR.w;

		Vec4f camTLDir = camTL - camPos;
		camTLDir = normalize(camTLDir);

		Vec4f camTRDir = camTR - camPos;
		camTRDir = normalize(camTRDir);

		Vec4f camBLDir = camBL - camPos;
		camBLDir = normalize(camBLDir);

		Vec4f camBRDir = camBR - camPos;
		camBRDir = normalize(camBRDir);

		cornerRays[0] = Ray4f(camPos, camTLDir);
		cornerRays[1] = Ray4f(camPos, camTRDir);
		cornerRays[2] = Ray4f(camPos, camBLDir);
		cornerRays[3] = Ray4f(camPos, camBRDir);

		rayI = camTRDir - camTLDir;
		rayJ = camBLDir - camTLDir;
	}

	Ray4f generateRay(float i, float j)
	{
		Vec4f org = getPosition();
		Vec4f dir = normalize(cornerRays[0].dir + rayI*i + rayJ*j);
		
		return Ray4f(org, dir);
	}

	// get camera world params
	Vec4f getRight() const
	{
		return Vec4f(worldMat.m11, worldMat.m12, worldMat.m13, worldMat.m14);
	}
	Vec4f getUp() const
	{
		return Vec4f(worldMat.m21, worldMat.m22, worldMat.m23, worldMat.m24);
	}
	Vec4f getDirection() const
	{
		return Vec4f(worldMat.m31, worldMat.m32, worldMat.m33, worldMat.m34);
	}
	Vec4f getPosition() const
	{
		return Vec4f(worldMat.m41, worldMat.m42, worldMat.m43, worldMat.m44);
	}
	
	// camera data planes
	float _leftPlane, _rightPlane, _bottomPlane, _topPlane, _nearPlane, _farPlane;

	// camera data rays
	Ray4f cornerRays[4];
	Vec4f rayI, rayJ;

	// camera data matrices
	Mat44f worldMat, viewMat, projMat;

	// temp, will be removed (camera data - ray frame)
	Vec4f origin;
	Vec4f view;
	Vec4f right;
	Vec4f up;
};

#endif
