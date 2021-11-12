#include "integrators/sure_based_integrator.h"
#include "lightdistrib.h"
#include "progressreporter.h"

namespace pbrt {
SUREBasedIntegrator *CreateSUREBasedIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int max_depth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixel_bounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixel_bounds = Intersect(pixel_bounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixel_bounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rr_threshold = params.FindOneFloat("rrthreshold", 1.);
    std::string light_strategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    PathIntegrator *path_integrator = new PathIntegrator(max_depth, camera, sampler, pixel_bounds,
                              rr_threshold, light_strategy);
    return new SUREBasedIntegrator(sampler, camera, std::make_shared<PathIntegrator>(path_integrator));
}

void SUREBasedIntegrator::Render(const Scene& scene) {
}

}  // namespace pbrt
