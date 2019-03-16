varying vec4 v_posW;
varying vec2 v_TexCoord;

uniform vec4 eyeW;
uniform sampler2D texNorm;

#define M_PI 3.1415926535897932384626433832795

vec4 HDR(vec4 rgb, float exposure)
{
	return vec4(1.0 - exp(-exposure*rgb.x), 1.0 - exp(-exposure*rgb.y), 1.0 - exp(-exposure*rgb.z), 0.0 );
}

// sky luminance distribution
float perez(float A, float B, float C, float D, float E, float theta, float gamma)
{
	return ( 1.0 + A*exp(B/cos(theta)) ) * ( 1.0 + C*exp(D*gamma) + E*cos(gamma)*cos(gamma) );
}

vec4 getSkyColor( vec4 sun_dir, vec4 view_dir, float T, float exposure, float sun_size, float sunIntensity )
{

	vec4 zenit = vec4(0.0, 1.0, 0.0, 0.0);
	float theta_sun =acos(clamp(dot(zenit,sun_dir), -0.99999, 0.99999));
	float theta = acos(clamp(dot(zenit,view_dir), -0.99999, 0.99999));
	float gamma = acos(clamp(dot(sun_dir,view_dir), -0.99999, 0.99999));

	vec4 A, B, C, D, E, zenith, xyY, XYZ;
	vec4 rgb;

	if (theta > M_PI/2.0-0.00001) theta = M_PI/2.0-0.00001;

	A.x = -0.0193*T - 0.2592; A.y = -0.0167*T - 0.2608; A.z =   0.1787*T - 1.4630;
	B.x = -0.0665*T + 0.0008; B.y = -0.0950*T + 0.0092; B.z =  -0.3554*T + 0.4275;
	C.x = -0.0004*T + 0.2125; C.y = -0.0079*T + 0.2102; C.z =  -0.0227*T + 5.3251;
	D.x = -0.0641*T - 0.8989; D.y = -0.0441*T - 1.6537; D.z =   0.1206*T - 2.5771;
	E.x = -0.0033*T + 0.0452; E.y = -0.0109*T + 0.0529; E.z =  -0.0670*T + 0.3703;

	float chi = (4.0/9.0 - T/120.0)*(M_PI - 2.0*theta_sun);
	zenith.z = ((4.0453*T - 4.9710)*tan(chi) - 0.2155*T + 2.4192) * 1000.0;
	zenith.x = ( 0.00166*theta_sun*theta_sun*theta_sun - 0.00375*theta_sun*theta_sun + 0.00209*theta_sun + 0.00000 )*T*T +
	           (-0.02903*theta_sun*theta_sun*theta_sun + 0.06377*theta_sun*theta_sun - 0.03202*theta_sun + 0.00394 )*T   +
			   ( 0.11693*theta_sun*theta_sun*theta_sun - 0.21196*theta_sun*theta_sun + 0.06052*theta_sun + 0.25886 );
	zenith.y = ( 0.00275*theta_sun*theta_sun*theta_sun - 0.00610*theta_sun*theta_sun + 0.00317*theta_sun + 0.00000 )*T*T +
	           (-0.04214*theta_sun*theta_sun*theta_sun + 0.08970*theta_sun*theta_sun - 0.04153*theta_sun + 0.00516 )*T   +
			   ( 0.15346*theta_sun*theta_sun*theta_sun - 0.26756*theta_sun*theta_sun + 0.06670*theta_sun + 0.26688 );
	
	xyY.x = zenith.x * (perez(A.x,B.x,C.x,D.x,E.x,theta,gamma)/perez(A.x,B.x,C.x,D.x,E.x,0.0,theta_sun));
	xyY.y = zenith.y * (perez(A.y,B.y,C.y,D.y,E.y,theta,gamma)/perez(A.y,B.y,C.y,D.y,E.y,0.0,theta_sun));
	xyY.z = zenith.z * (perez(A.z,B.z,C.z,D.z,E.z,theta,gamma)/perez(A.z,B.z,C.z,D.z,E.z,0.0,theta_sun));
	float Kn = zenith.z * (perez(A.z,B.z,C.z,D.z,E.z,theta_sun,0.0)/perez(A.z,B.z,C.z,D.z,E.z,0.0,theta_sun));
	xyY.z /= Kn;

	// convert xyY to XYZ
	XYZ.x = xyY.x/xyY.y*xyY.z;
	XYZ.y = xyY.z;
	XYZ.z = ((1.0 - xyY.x - xyY.y) / xyY.y) * xyY.z;
	
	// convert XYZ to rgb
	rgb.x =  3.240479*XYZ.x - 1.537150*XYZ.y - 0.498535*XYZ.z;
	rgb.y = -0.969256*XYZ.x + 1.875992*XYZ.y + 0.041556*XYZ.z;
	rgb.z =  0.055648*XYZ.x - 0.204043*XYZ.y + 1.057311*XYZ.z;
	rgb.w = XYZ.y;

	// sun highlighting
	if( degrees(gamma) < sun_size)
	{
		float sigma = 0.5;
		float x = degrees(gamma)/sun_size*3.0; 
		rgb.x += sunIntensity * exp( -(x*x)/(2.0*sigma*sigma) ) * rgb.x;
		rgb.y += sunIntensity * exp( -(x*x)/(2.0*sigma*sigma) ) * rgb.y;
		rgb.z += sunIntensity * exp( -(x*x)/(2.0*sigma*sigma) ) * rgb.z;
		rgb.w += sunIntensity * exp( -(x*x)/(2.0*sigma*sigma) ) * rgb.w;
	}
	rgb = HDR(rgb,exposure);
	return rgb;
}

void main()
{
	vec4 nI = normalize( vec4(0.5, 0.25, -0.75, 0.0) );
	vec4 eyeWVec = normalize(eyeW - v_posW);
	
	vec4 nS = vec4(0.0, 1.0, 0.0, 0.0);
	nS = 2.0 * texture2D(texNorm, v_TexCoord).yzxw - 1.0;
	nS.w = 0.0;
	
	vec4 reflect_vec = reflect(-nI, nS);
	
	vec4 upwelling = 1.25 * vec4(0, 0.15, 0.3, 0.0);
	vec4 sky = vec4(0.69, 0.84, 1.0, 0.0);
	vec4 air = 1.0 * vec4(0.1, 0.1, 0.1, 0.0);
	vec4 sun = vec4(1.0, 1.0, 0.6, 0.0);
	float nSnell = 1.34;
	float Kdiffuse = 0.91;
	
	float costhetai = abs(dot(nI,nS));
	float thetai = acos(costhetai);
	float sinthetat = sin(thetai + 0.5*3.14)/nSnell;
	float thetat = asin(sinthetat);
	float reflectivity;
	if(thetai == 0.0)
	{
		reflectivity = (nSnell - 1.0)/(nSnell + 1.0);
		reflectivity = reflectivity * reflectivity;
	}
	else
	{
		float fs = sin(thetat - thetai) / sin(thetat + thetai);
		float ts = tan(thetat - thetai) / tan(thetat + thetai);
		reflectivity = 0.5 * ( fs*fs + ts*ts );
	}
	
	float dist = length(eyeW - v_posW) * Kdiffuse * 0.00009;
	dist = exp(-dist);
	
	float cos_spec = clamp(dot(reflect_vec, eyeWVec), 0.0, 1.0);
	float pow_spec = pow(cos_spec, 450.0);
	
	sky = getSkyColor(nI, normalize(eyeW - v_posW), 2.25, 2.5, 5.0, 25.0);
	sky = sky * sky * 2.5;
	
	vec4 atmosphere = v_posW-eyeW;
	atmosphere.y = 0.0;
	atmosphere = normalize(atmosphere);
	air = getSkyColor(nI, atmosphere, 2.25, 2.5, 5.0, 25.0);
	
	gl_FragColor = dist * ( reflectivity * sky + (1.0-reflectivity) * upwelling ) + (1.0-dist)* air + sun * pow_spec;
	//gl_FragColor = vec4(1.0,0.0,0.0,1.0);
	
	//vec2 v_TexCoord2 = fract(v_TexCoord);
	//gl_FragColor = vec4(v_TexCoord2.x,v_TexCoord2.y,0.0,0.0);
}