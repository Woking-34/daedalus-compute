__constant float wave_a[2] = { 0.4f, 0.4f };
__constant float wave_k[2] = { 1.5f, 1.5f };
__constant float wave_w[2] = { 0.01f, 0.01f };
__constant float wave_p[2] = { 1.0f, 1.0f };
__constant float4 wave_dir = (float4)( 1.0f, 0.0f, -1.0f, 0.0f);

inline float4 generatePosition(
	float idX, float idY, int currDeform, float currTime,
	float gridWidth, float gridWidthDt, float gridLength, float gridLengthDt)
{
	float4 ret;
	
	float x = idX * gridWidthDt - gridWidth * 0.5f;
	float y = 0.0f;
	float z = idY * gridLengthDt - gridLength * 0.5f;
	
	// Directional
	if(currDeform == 0)
	{
		float dist = dot( wave_dir, (float4)(x, y, z, 0.0f) );
		for(int i = 0; i < 2; i++)
		{
			y += wave_a[i] * sin(wave_k[i]*dist - currTime*wave_w[i] + wave_p[i]);
		}
	}
	
	// Circular
	if(currDeform == 1)
	{
		float dist = sqrt(x*x + z*z);
		for(int i = 0; i < 2; i++)
		{
			y += wave_a[i] * sin(wave_k[i]*dist - currTime*wave_w[i] + wave_p[i]);
		}
	}
	
	// SinCos
	if(currDeform == 2)
	{
		y = cos(z + - 0.5f * currTime*wave_w[0]) + 0.5f * sin(x + 2.0f * z - 0.5f * currTime*wave_w[1]);
	}
	
	ret.x = x;
	ret.y = y;
	ret.z = z;
	ret.w = 1.0f;
	
	return ret;
}

__kernel
void deform(
	__global float4* restrict gridVerts, const int currDeform, const float currTime,
	const float gridWidth, const float gridWidthDt, const float gridLength, const float gridLengthDt)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	
	int sizeX = get_global_size(0);
	//int sizeY = get_global_size(1);
	
	float idXF = (float)(idX);
	float idYF = (float)(idY);
	
	float4 pos = generatePosition(idXF, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
	
	float4 normal;
	
	{
		float4 left = generatePosition(idXF-gridWidth, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 right = generatePosition(idXF+gridWidth, idYF, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 bottom = generatePosition(idXF, idYF-gridLengthDt, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		float4 top = generatePosition(idXF, idYF+gridLengthDt, currDeform, currTime, gridWidth, gridWidthDt, gridLength, gridLengthDt);
		
		float4 tangentX = right - left;
		
		if (dot(tangentX, tangentX) < 1e-10f)
			tangentX = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
		else
			tangentX = normalize(tangentX);
		
		float4 tangentY = bottom - top;
		
		if (dot(tangentY, tangentY) < 1e-10f)
			tangentY = (float4)(0.0f, 1.0f, 0.0f, 0.0f);
		else
			tangentY = normalize(tangentY);
		
		normal = cross(tangentX, tangentY);
		//normal = abs(normal);
	}
	
	gridVerts[2 * (idY * sizeX + idX) + 0] = pos;
	gridVerts[2 * (idY * sizeX + idX) + 1] = normal;
}