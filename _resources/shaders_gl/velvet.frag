uniform vec4 base_color;
uniform vec4 sheen;
uniform vec4 shiny;
uniform float roughness;
uniform float edginess;
uniform float backscatter;
uniform vec4 global_ambient;
uniform vec4 Ka;
uniform vec4 light_pos;
uniform vec4 light_color;
uniform vec4 Kd;
uniform vec4 eye_pos;


varying vec3 vPeye;
varying vec3 vNeye;



vec4 ambient(void)
{
   return global_ambient;
}

vec4 diffuse(vec3 Neye, vec3 Peye)
{
   // Compute normalized vector from vertex to light in eye space  (Leye)
   vec3 Leye = (vec3(light_pos) - Peye) / length(vec3(light_pos) - Peye);

   float NdotL = dot(Neye, Leye);

   // N.L
   return vec4(NdotL, NdotL, NdotL, NdotL);
}

void main(void)
{
	vec3 Nf = normalize(vNeye);           // Normalized normal vector
	vec3 Veye = -(vPeye / length(vPeye));  // Normalized eye vector
	
	if(dot(Nf,Veye)<0.0)
		Nf = -Nf;
	
	// For every light do the following:
	
	// Hemisphere
	vec3 Leye = ( vec3(light_pos) - vPeye) / length( vec3(light_pos) - vPeye);             // Leye for a given light
	
	// Retroreflective lobe
	float cosine = clamp(dot(Leye, Veye), 0.0, 1.0);
	
	vec4 LocalShiny = shiny + pow (cosine, 1.0 / roughness ) * backscatter * light_color * sheen;
	
	// Horizon scattering
	cosine     = clamp (dot(Nf, Veye), 0.0, 1.0);
	float sine = sqrt (1.0 - (cosine * cosine));
	LocalShiny += pow (sine, edginess) * dot(Leye, Nf) * light_color * sheen;
	
	// Uncomment one of these to see different components of illumination
	//gl_FragColor = Ka*ambient();
	//gl_FragColor = Kd*diffuse(Nf, vPeye);
	//gl_FragColor = shiny;
	
	// Add in diffuse color and return
	gl_FragColor = (Ka*ambient() + Kd*diffuse(Nf, vPeye)) * base_color + LocalShiny;
}