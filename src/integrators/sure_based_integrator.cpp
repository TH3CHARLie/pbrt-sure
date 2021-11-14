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
    PathIntegrator *path_integrator = new PathIntegrator(
        max_depth, camera, sampler, pixel_bounds, rr_threshold, light_strategy);

    int64_t sample_budget = params.FindOneInt("samplebudget", 64);
    int64_t num_initial_samples = params.FindOneInt("initialsamples", 8);
    return new SUREBasedIntegrator(
        sampler, camera, std::shared_ptr<PathIntegrator>(path_integrator),
        pixel_bounds, sample_budget, num_initial_samples);
}

void SUREBasedIntegrator::Render(const Scene &scene) {
    path_integrator->Preprocess(scene, *sampler);
    Bounds2i sample_bounds = camera->film->GetSampleBounds();
    Vector2i sample_extent = sample_bounds.Diagonal();
    const int tile_size = 16;
    Point2i num_tiles((sample_extent.x + tile_size - 1) / tile_size,
                      (sample_extent.y + tile_size - 1) / tile_size);

    ProgressReporter reporter(num_tiles.x * num_tiles.y,
                              "SURE-Based Integrator rendering");
    {
        ParallelFor2D(
            [&](Point2i tile) {
                MemoryArena arena;

                int seed = tile.y * num_tiles.x + tile.x;
                std::unique_ptr<Sampler> tile_sampler = sampler->Clone(seed);
                int x0 = sample_bounds.pMin.x + tile.x * tile_size;
                int x1 = std::min(x0 + tile_size, sample_bounds.pMax.x);
                int y0 = sample_bounds.pMin.y + tile.y * tile_size;
                int y1 = std::min(y0 + tile_size, sample_bounds.pMax.y);
                Bounds2i tile_bounds(Point2i(x0, y0), Point2i(x1, y1));

                std::unique_ptr<FilmTile> film_tile =
                    camera->film->GetFilmTile(tile_bounds);
                for (Point2i pixel : tile_bounds) {
                    tile_sampler->StartPixel(pixel);

                    if (!InsideExclusive(pixel, pixel_bounds)) {
                        continue;
                    }

                    do {
                        CameraSample camera_sample =
                            tile_sampler->GetCameraSample(pixel);

                        RayDifferential ray;
                        Float ray_weight = camera->GenerateRayDifferential(
                            camera_sample, &ray);
                        ray.ScaleDifferentials(
                            1 /
                            std::sqrt((Float)tile_sampler->samplesPerPixel));
                        Spectrum L(0.f);
                        SUREBasedAuxiliaryData auxiliary;
                        if (ray_weight > 0) {
                            L = path_integrator->Li_SURE_ext(ray, scene, *tile_sampler,
                                                    arena, auxiliary, 0);
                        }

                        if (L.HasNaNs()) {
                            L = Spectrum(0.f);
                        } else if (L.y() < -1e-5) {
                            L = Spectrum(0.f);
                        } else if (std::isinf(L.y())) {
                            L = Spectrum(0.f);
                        }
                        film_tile->AddSample_SURE_ext(camera_sample.pFilm, L, auxiliary,
                                             ray_weight);

                        arena.Reset();
                    } while (tile_sampler->StartNextSample());
                }
                camera->film->MergeFilmTile(std::move(film_tile));
                reporter.Update();
            },
            num_tiles);
        reporter.Done();
    }

    camera->film->WriteImage();
    camera->film->WriteTextureImage();
    camera->film->WriteNormalImage();
    camera->film->WriteDepthImage();
}

}  // namespace pbrt
