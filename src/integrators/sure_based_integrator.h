#ifndef SURE_BASED_INTEGRATOR_H
#define SURE_BASED_INTEGRATOR_H

#include "camera.h"
#include "integrator.h"
#include "integrators/path.h"
#include "pbrt.h"
#include "paramset.h"
#include "sampler.h"
#include <memory>

namespace pbrt {
class AdaptiveSampler : public Sampler {

};

class SUREBasedIntegrator : public Integrator {
  public:
    SUREBasedIntegrator(std::shared_ptr<Sampler> sampler,
                        std::shared_ptr<const Camera> camera,
                        std::shared_ptr<PathIntegrator> path_integrator,\
                        const Bounds2i pixel_bounds)
        : sampler(sampler), camera(camera), path_integrator(path_integrator), pixel_bounds(pixel_bounds) {}

    virtual ~SUREBasedIntegrator() {}

    virtual void Render(const Scene& scene);

  private:
    std::shared_ptr<Sampler> sampler;
    std::shared_ptr<const Camera> camera;
    std::shared_ptr<PathIntegrator> path_integrator;
    const Bounds2i pixel_bounds;
};

SUREBasedIntegrator *CreateSUREBasedIntegrator(
    const ParamSet &params,
    std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);
}  // namespace pbrt

#endif
