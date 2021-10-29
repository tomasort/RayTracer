////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <stack>

// Eigen for matrix operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
#include "utils.h"

// JSON parser library (https://github.com/nlohmann/json)
#include "json.hpp"
using json = nlohmann::json;

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;
int MAX_BOUNCE = 5;
bool SHOW_PROGRESS = false;
int MESH_INTERSECTION_METHOD = 2;

////////////////////////////////////////////////////////////////////////////////
// Define types & classes
////////////////////////////////////////////////////////////////////////////////

struct Ray {
	Vector3d origin;
	Vector3d direction;
	Ray() { }
	Ray(Vector3d o, Vector3d d) : origin(o), direction(d) { }
};

struct Light {
	Vector3d position;
	Vector3d intensity;
};

struct Intersection {
	Vector3d position;
	Vector3d normal;
	double ray_param;
};

struct Camera {
	bool is_perspective;
	Vector3d position;
	double field_of_view; // between 0 and PI
	double focal_length;
	double lens_radius; // for depth of field
};

struct Material {
	Vector3d ambient_color;
	Vector3d diffuse_color;
	Vector3d specular_color;
	double specular_exponent; // Also called "shininess"

	Vector3d reflection_color;
	Vector3d refraction_color;
	double refraction_index;
};

struct Object {
	Material material;
	virtual ~Object() = default; // Classes with virtual methods should have a virtual destructor!
	virtual bool intersect(const Ray &ray, Intersection &hit) = 0;
};

// We use smart pointers to hold objects as this is a virtual class
typedef std::shared_ptr<Object> ObjectPtr;

struct Sphere : public Object {
	Vector3d position;
	double radius;

	virtual ~Sphere() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Parallelogram : public Object {
	Vector3d origin;
	Vector3d u;
	Vector3d v;

	virtual ~Parallelogram() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct AABBTree {
	struct Node {
		AlignedBox3d bbox;
		int parent; // Index of the parent node (-1 for root)
		int left; // Index of the left child (-1 for a leaf)
		int right; // Index of the right child (-1 for a leaf)
		int triangle; // Index of the node triangle (-1 for internal nodes)
        int triangle_facet; // Index of the node triangle (-1 for internal nodes)
	};

	std::vector<Node*> nodes;
    std::vector<AlignedBox3d> bboxes;
    int root;

	AABBTree() = default; // Default empty constructor
	AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh
    int make_AABBTree(const MatrixXd &V, const MatrixXi &F, const MatrixXd &centroids, int parent_indx,
                      std::vector<int> &indices, int low, int high, const Vector3d &parent_box_size);
};

struct Mesh : public Object {
	MatrixXd vertices; // n x 3 matrix (n points)
	MatrixXi facets; // m x 3 matrix (m triangles)

	AABBTree bvh;

	Mesh() = default; // Default empty constructor
	Mesh(const std::string &filename);
	virtual ~Mesh() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
    bool intersect_helper(const Ray &ray, const AABBTree::Node &root, Intersection &hit, bool &found);
};

struct Scene {
	Vector3d background_color;
	Vector3d ambient_light;

	Camera camera;
	std::vector<Material> materials;
	std::vector<Light> lights;
	std::vector<ObjectPtr> objects;
};

////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, MatrixXd &V, MatrixXi &F) {
	std::ifstream in(filename);
	std::string token;
	in >> token;
	int nv, nf, ne;
	in >> nv >> nf >> ne;
	V.resize(nv, 3);
	F.resize(nf, 3);
	for (int i = 0; i < nv; ++i) {
		in >> V(i, 0) >> V(i, 1) >> V(i, 2);
	}
	for (int i = 0; i < nf; ++i) {
		int s;
		in >> s >> F(i, 0) >> F(i, 1) >> F(i, 2);
		assert(s == 3);
	}
}

Mesh::Mesh(const std::string &filename) {
	// Load a mesh from a file (assuming this is a .off file), and create a bvh
	load_off(filename, vertices, facets);
    bvh = AABBTree(vertices, facets);
}

////////////////////////////////////////////////////////////////////////////////
// BVH Implementation
////////////////////////////////////////////////////////////////////////////////

// Bounding box of a triangle
AlignedBox3d bbox_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
	AlignedBox3d box;
	box.extend(a);
	box.extend(b);
	box.extend(c);
	return box;
}

AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F) {
    // V: the vertices of the triangles in the mesh
    // F: the facets of the triangles in the mesh
    std::vector<int> indices;
	// Compute the centroids of all the triangles in the input mesh
	MatrixXd centroids(F.rows(), V.cols());
	centroids.setZero();
	for (int i = 0; i < F.rows(); ++i) {
	    indices.push_back(i); // This is just to create a vector of the indices for the recursive function
        for (int k = 0; k < F.cols(); ++k) {
			centroids.row(i) += V.row(F(i, k));
		}
		centroids.row(i) /= F.cols();
	}
    // Method (1): Top-down approach.
    // Split each set of primitives into 2 sets of roughly equal size, based on sorting the centroids.
    std::cout << "Making the top level box" << std::endl;
    AlignedBox3d bbox;
    for(int i = 0; i < F.rows(); i++){
        Vector3d a = V.row(F.row(i)[0]);
        Vector3d b = V.row(F.row(i)[1]);
        Vector3d c = V.row(F.row(i)[2]);
        AlignedBox3d box = bbox_triangle(a, b, c);
        bboxes.push_back(box);
        bbox.extend(box);
    }
    std::cout << "bboxes size: " << bboxes.size() << std::endl;
    Vector3d op = bbox.sizes();
    std::cout << "going into recursive calls:" << std::endl;
    root = make_AABBTree(V, F, centroids, -1, indices, 0, indices.size(), op);
    std::cout << "done with the recursive calls:" << std::endl;
    // Method (2): Bottom-up approach.
    // Merge nodes 2 by 2, starting from the leaves of the forest, until only 1 tree is left.
}

int AABBTree::make_AABBTree(const MatrixXd &V, const MatrixXi &F, const MatrixXd &centroids, int parent_indx, std::vector<int> &indices, int low, int high, const Vector3d &parent_box_size){
    /*
     * This Function returns the index of the root of the tree in the AABB nodes vector
     * We use the indices vector to maintain a list of indices to index into the facets matrix
     */
    Node *node = new Node();
    node->parent = parent_indx;
    node->triangle = -1;
    node->left = -1;
    node->right = -1;
    node->triangle_facet = -1;
    int current_indx = nodes.size();
    nodes.push_back(node);
    if(high-low == 1){
        node->triangle = current_indx;
        node->triangle_facet = indices[low];
        node->bbox = bbox_triangle(V.row(F.row(indices[low])[0]), V.row(F.row(indices[low])[1]), V.row(F.row(indices[low])[2]));
        return current_indx;
    }
    int longest_dim = 0;
    // Find the longest dimension of the bbox (more or less) of the parent
    for(int i = 0; i < 3; i++){
        if(parent_box_size[longest_dim] < parent_box_size[i]){
            longest_dim = i;
        }
    }
//    std::vector<int> elements;
//    for(std::vector<int>::iterator it = indices.begin()+low; it != indices.begin()+high; it++){
//        elements.push_back(*it);
//    }
    // We need to sort the centroids according to the longest dim to see where we need to partition the box
//    std::cout << "sorting the indices from " << low << " to " << high << std::endl;
    std::sort(indices.begin()+low,indices.begin()+high,[longest_dim, &centroids](int a, int b){
        return centroids.row(a)[longest_dim] < centroids.row(b)[longest_dim];
    });
    int left_end = low + std::ceil((high-low)/2.0);
    // now we need to divide the parent box according to mid
    Vector3d left_box_size = parent_box_size;
    left_box_size[longest_dim] = centroids.row(indices[left_end-1])[longest_dim] - centroids.row(indices[low])[longest_dim];
    Vector3d right_box_size = parent_box_size;
    right_box_size[longest_dim] = centroids.row(indices[high-1])[longest_dim] - centroids.row(indices[left_end])[longest_dim];
//    std::cout << "Going into left subtree" << std::endl;
    node->left = make_AABBTree(V, F, centroids, current_indx, indices, low, left_end, left_box_size);
//    std::cout << "Going into right subtree" << std::endl;
    node->right = make_AABBTree(V, F, centroids, current_indx, indices, left_end, high, right_box_size);
    node->bbox.extend(nodes[node->left]->bbox);
    node->bbox.extend(nodes[node->right]->bbox);
//    std::cout << "Current node is done" << std::endl;
    return current_indx;
}

////////////////////////////////////////////////////////////////////////////////

bool Sphere::intersect(const Ray &ray, Intersection &hit) {
	// calculate the intersection between ray and sphere
	Vector3d sphere_origin = position;
	double sphere_radius = radius;
	double a = ray.direction.dot(ray.direction);
	double b = (2.0*ray.direction).dot(ray.origin-sphere_origin);
	double c = (ray.origin-sphere_origin).dot(ray.origin-sphere_origin)-std::pow(sphere_radius, 2);
	std::vector<double> intersections;
	intersections.push_back(((-1)*b + sqrt(pow(b, 2) - 4*a*c))/2*a);
	intersections.push_back(((-1)*b - sqrt(pow(b, 2) - 4*a*c))/2*a);
	double closest = std::numeric_limits<double>::infinity();
	Vector3d p;
	for(double t : intersections){
	    if(isnan(t) || t < 0){
	        continue;
	    }
	    p = ray.origin + t*(ray.direction);  // Point of intersection
	    if((p-ray.origin).norm() < closest){
	        closest = (p-ray.origin).norm();
	    }
	}
	if(closest != std::numeric_limits<double>::infinity()){
	    hit.position = p;
	    hit.normal = p-position;
	    return true;
	}
	return false;
}

bool Parallelogram::intersect(const Ray &ray, Intersection &hit) {
	// calculate the intersection with a parallelogram
	Matrix3d M;
	M << -this->u, -this->v, ray.direction;
	Vector3d b = this->origin - ray.origin;
	Vector3d solution = M.colPivHouseholderQr().solve(b);
	if (solution[0] >= 0 && solution[1] >= 0 && solution[2] > 0 && solution[0] <= 1 && solution[1] <= 1) {
	    // The ray hit the parallelogram, compute the exact intersection point
	    hit.position = ray.origin + solution[2]*ray.direction;
	    // Compute normal at the intersection point
	    hit.normal = (this->u.cross(this->v)).normalized();
        if((ray.origin-hit.position).normalized().dot(hit.normal) < 0){
            hit.normal = (this->v.cross(this->u)).normalized();
        }
	    return true;
	}
	return false;
}

// -----------------------------------------------------------------------------

bool intersect_triangle(const Ray &ray, const Vector3d &a, const Vector3d &b, const Vector3d &c, Intersection &hit) {
	// Compute whether the ray intersects the given triangle.
	Vector3d u = c-a;
    Vector3d v = b-a;
    Matrix3d M;
    M << -u, -v, ray.direction;
    Vector3d _b = a - ray.origin;
    Vector3d solution = M.colPivHouseholderQr().solve(_b);
    if (solution[0] >= 0 && solution[1] >= 0 && solution[2] > 0 && solution[0] + solution[1] <= 1) {
        // The ray hit the parallelogram, compute the exact intersection point
        hit.position = ray.origin + solution[2]*ray.direction;
        // Compute normal at the intersection point
        hit.normal = (u.cross(v)).normalized();
        if((ray.origin-hit.position).normalized().dot(hit.normal) < 0){
            hit.normal = (v.cross(u)).normalized();
        }
        return true;
    }
	return false;
}

bool intersect_box(const Ray &ray, const AlignedBox3d &box, bool f) {
	// Compute whether the ray intersects the given box (aligned box).
	// There is no need to set the resulting normal and ray parameter, since
	// we are not testing with the real surface here anyway.
    double ep = 0.001;
    Vector3d t0 = (box.min()-ray.origin).cwiseQuotient(ray.direction);
    Vector3d t1 = (box.max()-ray.origin).cwiseQuotient(ray.direction);
    double tmin = t0.cwiseMin(t1).maxCoeff();
    double tmax = t1.cwiseMax(t0).minCoeff();
//    if(f){
//        std::cout << "hit with box tmin: " << tmin << " " << ray.origin + tmin*ray.direction << std::endl;
//        std::cout << "hit with box tmax: " << tmax << " " << ray.origin + tmax*ray.direction << std::endl;
//    }
    if((tmin > tmax || tmax < 0)){
        return false;
    }
//    if(tmax < 0 - epsilon){
//        return false;
//    }
//    if (tmin > tmax + epsilon){
//        return false;
//    }
//    if(tmin < 0 - epsilon){
//        return true;
//    }
    return true;
}

bool Mesh::intersect(const Ray &ray, Intersection &closest_hit) {
	//  Mesh Intersection Method (1): Traverse every triangle and return the closest hit.
	if(MESH_INTERSECTION_METHOD == 1){
        double closest_hit_distance = std::numeric_limits<double>::infinity();
        for(int i = 0; i < this->facets.rows(); i++){
            Vector3d a = this->vertices.row(this->facets.row(i)[0]);
            Vector3d b = this->vertices.row(this->facets.row(i)[1]);
            Vector3d c = this->vertices.row(this->facets.row(i)[2]);
            Intersection hit;
            if(intersect_triangle(ray, a, b, c, hit)){
//                bool f = intersect_box(ray, bbox_triangle(a, b, c), false);
//                if (!f){
//                    std::cout << "hit with triangle but not with box" << std::endl;
//                    bool f = intersect_box(ray, bbox_triangle(a, b, c), true);
//                    std::cout << "hit with triangle: " << hit.position << std::endl;
//                }
                double distance_from_origin = (hit.position-ray.origin).norm();
                if (distance_from_origin < closest_hit_distance){
                    closest_hit_distance = distance_from_origin;
                    closest_hit = hit;
                }
            }
        }
        if (closest_hit_distance < std::numeric_limits<double>::infinity()){
            return true;
	    }
	}else{
        // TODO: Mesh Intersection Method (2): Traverse the BVH tree and test the intersection with triangles at the leaf nodes that intersects the input ray.
        bool found = false;
        return intersect_helper(ray, *bvh.nodes[bvh.root], closest_hit, found);
	}
    return false;
}

bool Mesh::intersect_helper(const Ray &ray, const AABBTree::Node &root, Intersection &hit, bool &found) {
    // basically a preorder traversal
    if (root.triangle >= 0){
        Vector3d a = this->vertices.row(this->facets.row(root.triangle_facet)[0]);
        Vector3d b = this->vertices.row(this->facets.row(root.triangle_facet)[1]);
        Vector3d c = this->vertices.row(this->facets.row(root.triangle_facet)[2]);
        Intersection possible_new_hit;
        if(intersect_triangle(ray, a, b, c, possible_new_hit)){
            if(found){
                double distance_from_origin = (possible_new_hit.position-ray.origin).norm();
                double current_closest = (hit.position-ray.origin).norm();
                if (distance_from_origin < current_closest){
                    hit = possible_new_hit;
                }
            }else{
                hit = possible_new_hit;
                found = true;
            }
        }
        return found;
    }
    if(intersect_box(ray, root.bbox, false)){
        bool x = intersect_helper(ray, *bvh.nodes[root.left], hit, found);
        bool y = intersect_helper(ray, *bvh.nodes[root.right], hit, found);
        return x || y;
    }
    return false;
}


////////////////////////////////////////////////////////////////////////////////
// Define ray-tracing functions
////////////////////////////////////////////////////////////////////////////////

// Function declaration here (could be put in a header file)
Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &object, const Intersection &hit, int max_bounce);
Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit);
bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light);
Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce);

// -----------------------------------------------------------------------------

Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &obj, const Intersection &hit, int max_bounce) {
	// Material for hit object
	const Material &mat = obj.material;

	// Ambient light contribution
	Vector3d ambient_color = obj.material.ambient_color.array() * scene.ambient_light.array();

	// Punctual lights contribution (direct lighting)
	Vector3d lights_color(0, 0, 0);
	for (const Light &light : scene.lights) {
		Vector3d Li = (light.position - hit.position).normalized();
		Vector3d N = hit.normal;

		// (shadow rays) We need a ray from the point of intersection to the light
//		Ray r;
//		r.origin = hit.position + hit.normal*0.01;
//        r.direction = (light.position- hit.position);
//        if (is_light_visible(scene, r, light)){
//		    // The ray did not intersect with any object on its way to the light so it is in shadows
//		    // In this case the light should not contribute to the color.
//		    continue;
//		}

		// Diffuse contribution
		Vector3d diffuse = mat.diffuse_color * std::max(Li.dot(N), 0.0);

		// TODO (Assignment 2, specular contribution)
		Vector3d specular(0, 0, 0);

		// Attenuate lights according to the squared distance to the lights
		Vector3d D = light.position - hit.position;
		lights_color += (diffuse + specular).cwiseProduct(light.intensity) /  D.squaredNorm();
	}

	// TODO (Assignment 2, reflected ray) (not required for assignment 4)
	Vector3d reflection_color(0, 0, 0);

	// TODO (Assignment 2, refracted ray) (not required for assignment 4)
	Vector3d refraction_color(0, 0, 0);

	// Rendering equation
	Vector3d C = ambient_color + lights_color + reflection_color + refraction_color;

	return C;
}

// -----------------------------------------------------------------------------

Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit) {
	int closest_index = -1;
	// find nearest hit
    int i = 0;
    for (ObjectPtr obj : scene.objects){
        Intersection hit;
	    if(obj->intersect(ray, hit)){
	        if(i == 0 || (hit.position-ray.origin).norm() < (closest_hit.position-ray.origin).norm()){
	            closest_hit = hit;
                closest_index = i;
	        }
	    }
	    i++;
	}
	if (closest_index < 0) {
		// Return a NULL pointer
		return nullptr;
	} else {
		// Return a pointer to the hit object. Don't forget to set 'closest_hit' accordingly!
		return scene.objects[closest_index].get();
	}
}

bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light) {
	Intersection i;
	if (find_nearest_object(scene, ray, i) == nullptr){
	    return true;
	}
	return false;
}

Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce) {
	Intersection hit;
	if (Object * obj = find_nearest_object(scene, ray, hit)) {
		// 'obj' is not null and points to the object of the scene hit by the ray
		return ray_color(scene, ray, *obj, hit, max_bounce);
	} else {
		// 'obj' is null, we must return the background color
		return scene.background_color;
	}
}

////////////////////////////////////////////////////////////////////////////////

void render_scene(const Scene &scene) {
	std::cout << "Simple ray tracer." << std::endl;

	int w = 640;
	int h = 480;
	MatrixXd R = MatrixXd::Zero(w, h);
	MatrixXd G = MatrixXd::Zero(w, h);
	MatrixXd B = MatrixXd::Zero(w, h);
	MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

	// The camera always points in the direction -z
	// The sensor grid is at a distance 'focal_length' from the camera center,
	// and covers an viewing angle given by 'field_of_view'.
	double aspect_ratio = double(w) / double(h);
	double scale_y = 1.0; // TODO: (field of view) Stretch the pixel grid by the proper amount here to produce the field of view (not required for assignment 4)
	double scale_x = 1.0; //

	// The pixel grid through which we shoot rays is at a distance 'focal_length'
	// from the sensor, and is scaled from the canonical [-1,1] in order
	// to produce the target field of view.
	Vector3d grid_origin(-scale_x, scale_y, -scene.camera.focal_length);
    Vector3d x_displacement((2.0/w*scale_x), 0, 0);
    Vector3d y_displacement(0, (-2.0/h*scale_y)*aspect_ratio, 0);
    if(w > h){
        x_displacement = Vector3d((2.0/w*scale_x)*aspect_ratio, 0, 0);
        y_displacement = Vector3d(0, (-2.0/h*scale_y), 0);
    }
	for (unsigned i = 0; i < w; ++i) {
	    if (SHOW_PROGRESS){
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Ray tracing: " << (100.0 * i) / w << "%\r" << std::flush;
	    }
		for (unsigned j = 0; j < h; ++j) {
			// TODO (Assignment 2, depth of field) (not required for assignment 4)
			Vector3d shift = grid_origin + (i+0.5)*x_displacement + (j+0.5)*y_displacement;

			// Prepare the ray
			Ray ray;
			if (scene.camera.is_perspective) {
				// Perspective camera
				ray.origin = scene.camera.position;
                ray.direction =  (shift - ray.origin).normalized();
			} else {
				// Orthographic camera
				ray.origin = scene.camera.position + Vector3d(shift[0], shift[1], 0);
				ray.direction = Vector3d(0, 0, -1);
			}

			Vector3d C = shoot_ray(scene, ray, MAX_BOUNCE);
			R(i, j) = C(0);
			G(i, j) = C(1);
			B(i, j) = C(2);
			A(i, j) = 1;
		}
	}
    if (SHOW_PROGRESS){
        std::cout << "Ray tracing: 100%  " << std::endl;
    }

	// Save to png
	const std::string filename("raytrace.png");
	write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

Scene load_scene(const std::string &filename) {
	Scene scene;

	// Load json data from scene file
	json data;
	std::ifstream in(filename);
	in >> data;

	// Helper function to read a Vector3d from a json array
	auto read_vec3 = [] (const json &x) {
		return Vector3d(x[0], x[1], x[2]);
	};

	// Read scene info
	scene.background_color = read_vec3(data["Scene"]["Background"]);
	scene.ambient_light = read_vec3(data["Scene"]["Ambient"]);

	// Read camera info
	scene.camera.is_perspective = data["Camera"]["IsPerspective"];
	scene.camera.position = read_vec3(data["Camera"]["Position"]);
	scene.camera.field_of_view = data["Camera"]["FieldOfView"];
	scene.camera.focal_length = data["Camera"]["FocalLength"];
	scene.camera.lens_radius = data["Camera"]["LensRadius"];

	// Read materials
	for (const auto &entry : data["Materials"]) {
		Material mat;
		mat.ambient_color = read_vec3(entry["Ambient"]);
		mat.diffuse_color = read_vec3(entry["Diffuse"]);
		mat.specular_color = read_vec3(entry["Specular"]);
		mat.reflection_color = read_vec3(entry["Mirror"]);
		mat.refraction_color = read_vec3(entry["Refraction"]);
		mat.refraction_index = entry["RefractionIndex"];
		mat.specular_exponent = entry["Shininess"];
		scene.materials.push_back(mat);
	}

	// Read lights
	for (const auto &entry : data["Lights"]) {
		Light light;
		light.position = read_vec3(entry["Position"]);
		light.intensity = read_vec3(entry["Color"]);
		scene.lights.push_back(light);
	}

	// Read objects
	for (const auto &entry : data["Objects"]) {
		ObjectPtr object;
		if (entry["Type"] == "Sphere") {
			auto sphere = std::make_shared<Sphere>();
			sphere->position = read_vec3(entry["Position"]);
			sphere->radius = entry["Radius"];
			object = sphere;
		} else if (entry["Type"] == "Parallelogram") {
		    auto parallelogram = std::make_shared<Parallelogram>();
		    parallelogram->origin = read_vec3(entry["origin"]);
		    parallelogram->u = read_vec3(entry["u"]);
            parallelogram->v = read_vec3(entry["v"]);
		} else if (entry["Type"] == "Mesh") {
			// Load mesh from a file
            std::string DATA_DIR = "../data/"; // I think is better to have this either as an argument or in the scene.json
            std::string filename = std::string(DATA_DIR) + entry["Path"].get<std::string>();
			object = std::make_shared<Mesh>(filename);
		}
		object->material = scene.materials[entry["Material"]];
		scene.objects.push_back(object);
	}

	return scene;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " scene.json" << std::endl;
		return 1;
	}
	Scene scene = load_scene(argv[1]);
	render_scene(scene);
	return 0;
}
