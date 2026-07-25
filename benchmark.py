from matplotlib import pyplot
from renderer import *
from time import time
from lights import *
import numpy as np
import json
import os
import statistics
from datetime import datetime

# ——————————[ CONFIG ]——————————————————————————————————————————————————————
N_RUNS   = 4
OUT_DIR  = "benchmark_output"
LOG_FILE = os.path.join(OUT_DIR, "benchmark_log.json")
MD_FILE  = os.path.join(OUT_DIR, "benchmark_report.md")
os.makedirs(OUT_DIR, exist_ok=True)

# ——————————[ ASSETS ]——————————————————————————————————————————————————————
start_time = time()
grid_atlas       = Atlas(pyplot.imread("grid.png"))

utah_teapot      = load_obj('models/utah_teapot.obj',      smooth_shading=True)
utah_teapot_nss  = load_obj('models/utah_teapot.obj',      smooth_shading=False)
utah_teapot1     = load_obj('models/utah_teapot_reso1.obj', smooth_shading=True)
utah_teapot1_nss = load_obj('models/utah_teapot_reso1.obj', smooth_shading=False)
utah_teapot2     = load_obj('models/utah_teapot_reso2.obj', smooth_shading=True)
utah_teapot2_nss = load_obj('models/utah_teapot_reso2.obj', smooth_shading=False)
utah_teapot3     = load_obj('models/utah_teapot_reso3.obj', smooth_shading=True)
utah_teapot3_nss = load_obj('models/utah_teapot_reso3.obj', smooth_shading=False)

# ——————————[ CAMERAS ]—————————————————————————————————————————————————————
camera = Camera(
    width=400, height=400,
    pos=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
    d=300, view_dist=500, gamma=1.5
)
hd = (800, 800)

# ——————————[ LIGHTS ]——————————————————————————————————————————————————————
lights  = [PointLight(np.array([5, 5, 0]))]
lights1 = [
    PointLight(np.array([5, 5, 0])),
    PointLight(np.array([-5, 0, 0]),
               np.array([0.05, 0.01, 0.03]),
               np.array([0.5,  0.1,  0.3 ]),
               np.array([0.5,  0.1,  0.3 ]),
               intensity=0.4),
]
lights2 = [
    PointLight(np.array([5, 5, 0])),
    PointLight(np.array([-5, 0, 0]),
               np.array([0.05, 0.01, 0.03]),
               np.array([0.5,  0.1,  0.3 ]),
               np.array([0.5,  0.1,  0.3 ]),
               intensity=0.4),
    PointLight(np.array([-5, 5, 5]),
               np.array([0.01, 0.4,  0.06]),
               np.array([0.01, 0.4,  0.06]),
               np.array([0.01, 0.4,  0.06]),
               intensity=0.4),
]

# ——————————[ SHADER FACTORIES ]————————————————————————————————————————————
def make_grid_shader(lights_):
    return LitTexture(
        grid_atlas, lights_, True,
        ambient=np.array([0.05, 0.05, 0.05]),
        diffuse=np.array([0.7,  0.7,  0.7 ]),
        specular=np.array([1.0,  1.0,  1.0 ]),
        shininess=50, reflection=1,
    )

def make_flat_shader(lights_):
    return LitMaterial(
        lights_,
        ambient=np.array([0.05, 0.05, 0.05]),
        diffuse=np.array([0.7,  0.7,  0.7 ]),
        specular=np.array([1.0,  1.0,  1.0 ]),
        shininess=50, reflection=1,
    )

# ——————————[ INSTANCE HELPER ]—————————————————————————————————————————————
def inst(model, shader, pos=(0, -2, 5.5), rot=(-100, -12, 0), scale=(1, 1, 1)):
    return make_instance(
        model,
        Transform(Point3D(*pos), Point3D(*scale), Point3D(*rot)),
        shader,
    )

# ——————————[ TIMED RENDER ]————————————————————————————————————————————————
def timed_render(cam, lights_, objects, width, height):
    post_process = [PPFog(1200, [0, 0, 0])]
    viewport = Viewport(camera=cam, objects=objects, project_func=perspective_projection)

    # Count scene-level stats before rendering
    instance_list = []
    for instance in viewport.objects:
        if type(instance) is ComplexInstance:
            for inst_ in instance.instances:
                instance_list.append({
                    "triangles": len(inst_.model.triangles),
                    "shader":    type(inst_.mat).__name__,
                })
        else:
            instance_list.append({
                "triangles": len(instance.model.triangles),
                "shader":    type(instance.mat).__name__,
            })

    total_triangles = sum(i["triangles"] for i in instance_list)

    # Dict filled in by render
    benchmark_data = {}

    image, data_buffer = render(
        viewport, width, height,
        post_process=post_process,
        benchmark_data=benchmark_data,
    )

    # Extra stats derived from the finished data_buffer
    depth_channel  = data_buffer[:, :, 3]
    visible_pixels = depth_channel[depth_channel > 0]

    log = {
        # Timing (all in seconds)
        "time_pass1_geometry_s":    benchmark_data.get("time_pass1", None),
        "time_pass2_shading_s":     benchmark_data.get("time_pass2", None),
        "time_pass3_postprocess_s": benchmark_data.get("time_pass3", None),
        "time_total_s":             benchmark_data.get("time_total", None),

        # Scene complexity
        "n_instances":      len(instance_list),
        "instances":        instance_list,
        "total_triangles":  total_triangles,

        # G-buffer stats
        "pixels_total":          width * height,
        "pixels_shaded":         benchmark_data.get("pixels_shaded", 0),
        "pixels_shaded_pct":     round(benchmark_data.get("pixels_shaded", 0) / (width * height) * 100, 2),
        "depth_min":             round(float(visible_pixels.min()), 4) if len(visible_pixels) else None,
        "depth_max":             round(float(visible_pixels.max()), 4) if len(visible_pixels) else None,
        "depth_mean":            round(float(visible_pixels.mean()), 4) if len(visible_pixels) else None,

        # Throughput
        "triangles_per_second":  round(total_triangles / benchmark_data["time_pass1"], 0) if benchmark_data.get("time_pass1") else None,
        "pixels_per_second":     round(benchmark_data.get("pixels_shaded", 0) / benchmark_data["time_pass2"], 0) if benchmark_data.get("time_pass2") else None,
    }

    print(
        f"  [bench] total={log['time_total_s']:.3f}s  "
        f"p1={log['time_pass1_geometry_s']:.3f}s  "
        f"p2={log['time_pass2_shading_s']:.3f}s  "
        f"p3={log['time_pass3_postprocess_s']:.3f}s  "
        f"tri={total_triangles:,}  "
        f"px={log['pixels_shaded']:,} ({log['pixels_shaded_pct']}%)  "
        f"tri/s={log['triangles_per_second']:,.0f}  "
        f"px/s={log['pixels_per_second']:,.0f}"
    )
    return image, data_buffer, log

# ——————————[ SCENE DEFINITIONS ]———————————————————————————————————————────
SCENES = [

    # ── Shader type comparison — base resolution, 1 light ─────────────────
    {
        "name":        "Single teapot · base reso · smooth normals · grid texture · 1 light",
        "slug":        "01_teapot_base_smooth_grid_1light",
        "description": "Baseline: one teapot at standard resolution with smooth normals and a grid texture shader, single point light.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · smooth normals · flat colour · 1 light",
        "slug":        "02_teapot_base_smooth_flat_1light",
        "description": "Same geometry as 01 but with a flat LitMaterial instead of a texture, to isolate texture-sampling cost.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot, make_flat_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · flat shading (no smooth normals) · grid texture · 1 light",
        "slug":        "03_teapot_base_nss_grid_1light",
        "description": "Same geometry and texture as 01 but with flat (per-face) normals, to isolate the cost of normal interpolation.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot_nss, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · flat shading (no smooth normals) · flat colour · 1 light",
        "slug":        "04_teapot_base_nss_flat_1light",
        "description": "Cheapest possible configuration: flat normals, flat colour material, single light.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot_nss, make_flat_shader(l))],
    },

    # ── Light count comparison ─────────────────────────────────────────────
    {
        "name":        "Single teapot · base reso · smooth normals · grid texture · 2 lights",
        "slug":        "05_teapot_base_smooth_grid_2lights",
        "description": "Adds a second coloured point light to measure per-light shading overhead.",
        "camera": camera, "lights": lights1, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · smooth normals · grid texture · 3 lights",
        "slug":        "06_teapot_base_smooth_grid_3lights",
        "description": "Three point lights (white + pink + green) to measure full multi-light overhead.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · smooth normals · flat colour · 3 lights",
        "slug":        "07_teapot_base_smooth_flat_3lights",
        "description": "Three lights with flat colour material: isolates light-count cost without texture sampling.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot, make_flat_shader(l))],
    },
    {
        "name":        "Single teapot · base reso · flat shading · flat colour · 3 lights",
        "slug":        "08_teapot_base_nss_flat_3lights",
        "description": "Flat normals + flat colour + 3 lights: measures light-count cost with no interpolation overhead.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot_nss, make_flat_shader(l))],
    },

    # ── Reso1 mesh ─────────────────────────────────────────────────────────
    {
        "name":        "Single teapot · reso1 mesh · smooth normals · grid texture · 1 light",
        "slug":        "09_teapot_reso1_smooth_grid_1light",
        "description": "Higher-resolution mesh with smooth normals and grid texture, single light.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · smooth normals · flat colour · 1 light",
        "slug":        "10_teapot_reso1_smooth_flat_1light",
        "description": "Reso1 mesh, smooth normals, flat colour: isolates geometry + interpolation cost without texture.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2, make_flat_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · flat shading · grid texture · 1 light",
        "slug":        "11_teapot_reso1_nss_grid_1light",
        "description": "Reso1 mesh, flat normals, grid texture: high triangle count with no interpolation overhead.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2_nss, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · flat shading · flat colour · 1 light",
        "slug":        "12_teapot_reso1_nss_flat_1light",
        "description": "Cheapest reso1 configuration: flat normals, flat colour, single light.",
        "camera": camera, "lights": lights, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2_nss, make_flat_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · smooth normals · grid texture · 3 lights",
        "slug":        "13_teapot_reso1_smooth_grid_3lights",
        "description": "Most expensive single-teapot config: reso1 + smooth normals + grid texture + 3 lights.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2, make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · flat shading · flat colour · 3 lights",
        "slug":        "14_teapot_reso1_nss_flat_3lights",
        "description": "Reso1 mesh with minimum shading overhead and 3 lights.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [inst(utah_teapot2_nss, make_flat_shader(l))],
    },

    # ── Multi-instance — same mesh ─────────────────────────────────────────
    {
        "name":        "3× teapots · reso1 mesh · all smooth normals · grid texture · 3 lights",
        "slug":        "15_x3_reso1_all_smooth_grid_3lights",
        "description": "Three reso1 teapots side-by-side, all smooth + grid texture + 3 lights.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot1,     make_grid_shader(l), pos=(-1, -2, 5.5)),
            inst(utah_teapot2,     make_grid_shader(l), pos=( 0, -2, 5.5)),
            inst(utah_teapot3,     make_grid_shader(l), pos=( 1, -2, 5.5)),
        ],
    },
    {
        "name":        "3× teapots · reso1 mesh · all flat shading · flat colour · 3 lights",
        "slug":        "16_x3_reso1_all_nss_flat_3lights",
        "description": "Three reso1 teapots, cheapest shading: flat normals, flat colour, 3 lights.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot1_nss, make_flat_shader(l), pos=(-1, -2, 5.5)),
            inst(utah_teapot2_nss, make_flat_shader(l), pos=( 0, -2, 5.5)),
            inst(utah_teapot3_nss, make_flat_shader(l), pos=( 1, -2, 5.5)),
        ],
    },

    # ── Multi-instance — mixed mesh resolution ─────────────────────────────
    {
        "name":        "2× teapots · mixed reso (base + reso1) · both smooth normals · grid texture · 3 lights",
        "slug":        "17_mixed_reso_both_smooth_grid_3lights",
        "description": "One base-reso and one reso1 teapot, both smooth + grid. Measures effect of mixed triangle counts.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot,  make_grid_shader(l), pos=( 1, -2, 5.5)),
            inst(utah_teapot2, make_grid_shader(l), pos=(-1, -2, 5.5)),
        ],
    },
    {
        "name":        "2× teapots · mixed reso (base smooth + reso1 flat shading) · grid texture · 3 lights",
        "slug":        "18_mixed_reso_smooth_vs_nss_grid_3lights",
        "description": "Base-reso smooth next to reso1 flat: isolates interaction between mesh resolution and shading mode.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot,      make_grid_shader(l), pos=( 1, -2, 5.5)),
            inst(utah_teapot2_nss, make_grid_shader(l), pos=(-1, -2, 5.5)),
        ],
    },
    {
        "name":        "2× teapots · mixed reso (base smooth + reso1 flat shading) · flat colour · 3 lights",
        "slug":        "19_mixed_reso_smooth_vs_nss_flat_3lights",
        "description": "Same reso/shading mix as 18 but with flat colour material to remove texture cost.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot,      make_flat_shader(l), pos=( 1, -2, 5.5)),
            inst(utah_teapot2_nss, make_flat_shader(l), pos=(-1, -2, 5.5)),
        ],
    },
    {
        "name":        "3× teapots · mixed reso · mixed shading (base smooth grid + reso1 smooth flat + reso1 flat flat) · 3 lights",
        "slug":        "20_x3_mixed_reso_mixed_shading_3lights",
        "description": "Three teapots each with a different reso/shading/shader combination. Stress-tests the material registry.",
        "camera": camera, "lights": lights2, "resolution": (camera.width, camera.height),
        "build":  lambda l: [
            inst(utah_teapot,      make_grid_shader(l), pos=( 1, -2, 5.5)),
            inst(utah_teapot2,     make_flat_shader(l), pos=( 0, -2, 5.5)),
            inst(utah_teapot2_nss, make_flat_shader(l), pos=(-1, -2, 5.5)),
        ],
    },

    # ── HD resolution (800×800) ────────────────────────────────────────────
    {
        "name":        "Single teapot · base reso · smooth normals · grid texture · 3 lights · 800×800",
        "slug":        "21_teapot_base_smooth_grid_3lights_HD800",
        "description": "Scene 06 rerun at 800×800 to measure framebuffer-size impact on shading pass.",
        "camera": camera, "lights": lights2, "resolution": hd,
        "build":  lambda l: [inst(utah_teapot,  make_grid_shader(l))],
    },
    {
        "name":        "Single teapot · reso1 mesh · smooth normals · grid texture · 3 lights · 800×800",
        "slug":        "22_teapot_reso1_smooth_grid_3lights_HD800",
        "description": "Scene 13 rerun at 800×800: highest geometry + shading load at HD resolution.",
        "camera": camera, "lights": lights2, "resolution": hd,
        "build":  lambda l: [inst(utah_teapot2, make_grid_shader(l))],
    },
    {
        "name":        "3× teapots · reso1 mesh · all smooth normals · grid texture · 3 lights · 800×800",
        "slug":        "23_x3_reso1_smooth_grid_3lights_HD800",
        "description": "Scene 15 rerun at 800×800: absolute worst-case load.",
        "camera": camera, "lights": lights2, "resolution": hd,
        "build":  lambda l: [
            inst(utah_teapot1, make_grid_shader(l), pos=(-1, -2, 5.5)),
            inst(utah_teapot2, make_grid_shader(l), pos=( 0, -2, 5.5)),
            inst(utah_teapot3, make_grid_shader(l), pos=( 1, -2, 5.5)),
        ],
    },
]

# ——————————[ HELPERS ]—————————————————————————————————————————————————————
def stats(arr):
    return {
        "min":    round(min(arr), 4),
        "max":    round(max(arr), 4),
        "mean":   round(statistics.mean(arr), 4),
        "median": round(statistics.median(arr), 4),
        "stdev":  round(statistics.stdev(arr), 4) if len(arr) > 1 else 0.0,
    }

def fmt(s):
    return (f"mean {s['mean']:.3f}s  med {s['median']:.3f}s  "
            f"min {s['min']:.3f}s  max {s['max']:.3f}s  σ {s['stdev']:.3f}s")

# ——————————[ BENCHMARK LOOP ]——————————————————————————————————————————————
all_results = {
    "meta": {
        "date":          datetime.now().isoformat(),
        "n_runs":        N_RUNS,
        "resolution_sd": f"{camera.width}x{camera.height}",
        "resolution_hd": f"{hd[0]}x{hd[1]}",
    },
    "scenes": {}
}

for scene in SCENES:
    slug          = scene["slug"]
    name          = scene["name"]
    desc          = scene["description"]
    cam           = scene["camera"]
    lights_       = scene["lights"]
    width, height = scene["resolution"]
    build         = scene["build"]

    print(f"\n{'='*60}")
    print(f"SCENE: {name}")
    print(f"       ({N_RUNS} runs · {width}x{height})")
    print(f"{'='*60}")

    run_logs   = []
    t_total    = []
    t_pass1    = []
    t_pass2    = []
    t_pass3    = []
    saved_image = None

    for run in range(N_RUNS):
        print(f"\n  -- Run {run + 1}/{N_RUNS} --")
        objects = build(lights_)
        image, data_buffer, log = timed_render(cam, lights_, objects, width, height)

        run_logs.append(log)
        t_total.append(log["time_total_s"])
        t_pass1.append(log["time_pass1_geometry_s"])
        t_pass2.append(log["time_pass2_shading_s"])
        t_pass3.append(log["time_pass3_postprocess_s"])

        if saved_image is None:
            saved_image = image
            img_path = os.path.join(OUT_DIR, f"{slug}.png")
            pyplot.imsave(img_path, np.clip(saved_image, 0, 1))
            print(f"  [IMG] Saved → {img_path}")

    scene_result = {
        "name":        name,
        "description": desc,
        "image_saved": f"{slug}.png",
        "resolution":  f"{width}x{height}",
        "n_lights":    len(lights_),
        # Scene complexity (from first run — identical across runs)
        "n_instances":     run_logs[0]["n_instances"],
        "instances":       run_logs[0]["instances"],
        "total_triangles": run_logs[0]["total_triangles"],
        # G-buffer stats (from first run)
        "pixels_total":      run_logs[0]["pixels_total"],
        "pixels_shaded":     run_logs[0]["pixels_shaded"],
        "pixels_shaded_pct": run_logs[0]["pixels_shaded_pct"],
        "depth_min":         run_logs[0]["depth_min"],
        "depth_max":         run_logs[0]["depth_max"],
        "depth_mean":        run_logs[0]["depth_mean"],
        # Per-run raw logs
        "runs": run_logs,
        # Aggregated timing stats
        "stats": {
            "total_time_s":    stats(t_total),
            "pass1_time_s":    stats(t_pass1),
            "pass2_time_s":    stats(t_pass2),
            "pass3_time_s":    stats(t_pass3),
            # Throughput (averaged across runs)
            "triangles_per_second": round(statistics.mean(
                [r["triangles_per_second"] for r in run_logs if r["triangles_per_second"]]), 0),
            "pixels_per_second": round(statistics.mean(
                [r["pixels_per_second"] for r in run_logs if r["pixels_per_second"]]), 0),
        },
    }
    all_results["scenes"][slug] = scene_result

    print(f"\n  ── Summary ──")
    for k in ("pass1_time_s", "pass2_time_s", "pass3_time_s", "total_time_s"):
        print(f"    {k:22s}  {fmt(scene_result['stats'][k])}")
    print(f"    {'tri/s':22s}  {scene_result['stats']['triangles_per_second']:,.0f}")
    print(f"    {'px/s':22s}  {scene_result['stats']['pixels_per_second']:,.0f}")

# ——————————[ SAVE JSON ]———————————————————————————————————————————————————
with open(LOG_FILE, "w") as f:
    json.dump(all_results, f, indent=2)

# ——————————[ SAVE MARKDOWN ]———————————————————————————————————————————————
meta = all_results["meta"]
lines = [
    "# Renderer Benchmark Report",
    "",
    f"**Date:** {meta['date']}  ",
    f"**Runs per scene:** {meta['n_runs']}  ",
    f"**Resolution SD:** {meta['resolution_sd']}  ",
    f"**Resolution HD:** {meta['resolution_hd']}  ",
    f"**Total benchmark wall time:** {time() - start_time:.1f}s  ",
    "",
    "---",
    "",
]

groups = {
    "Shader type comparison — base resolution, 1 light":        ["01","02","03","04"],
    "Light count comparison — base resolution, smooth normals": ["05","06","07","08"],
    "Reso1 mesh — shader × shading × light count":              ["09","10","11","12","13","14"],
    "Multi-instance — same mesh resolution":                    ["15","16"],
    "Multi-instance — mixed mesh resolution":                   ["17","18","19","20"],
    "HD resolution (800×800)":                                  ["21","22","23"],
}

for group_title, prefixes in groups.items():
    lines += [f"## {group_title}", ""]
    for slug, sr in all_results["scenes"].items():
        if slug[:2] not in prefixes:
            continue
        s = sr["stats"]
        lines += [
            f"### {sr['name']}",
            "",
            f"> {sr['description']}",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| Resolution | {sr['resolution']} |",
            f"| Instances | {sr['n_instances']} |",
            f"| Lights | {sr['n_lights']} |",
            f"| Total triangles | {sr['total_triangles']:,} |",
            f"| Pixels total | {sr['pixels_total']:,} |",
            f"| Pixels shaded | {sr['pixels_shaded']:,} ({sr['pixels_shaded_pct']}%) |",
            f"| Depth range | {sr['depth_min']} → {sr['depth_max']} (mean {sr['depth_mean']}) |",
            f"| Avg throughput | {s['triangles_per_second']:,.0f} tri/s · {s['pixels_per_second']:,.0f} px/s |",
            f"| Image | ![]({sr['image_saved']}) |",
            "",
            "| Pass | Mean | Median | Min | Max | σ |",
            "|---|---|---|---|---|---|",
            f"| **Pass 1** — Geometry    | {s['pass1_time_s']['mean']:.3f}s | {s['pass1_time_s']['median']:.3f}s | {s['pass1_time_s']['min']:.3f}s | {s['pass1_time_s']['max']:.3f}s | {s['pass1_time_s']['stdev']:.3f}s |",
            f"| **Pass 2** — Shading     | {s['pass2_time_s']['mean']:.3f}s | {s['pass2_time_s']['median']:.3f}s | {s['pass2_time_s']['min']:.3f}s | {s['pass2_time_s']['max']:.3f}s | {s['pass2_time_s']['stdev']:.3f}s |",
            f"| **Pass 3** — Post/Gamma  | {s['pass3_time_s']['mean']:.3f}s | {s['pass3_time_s']['median']:.3f}s | {s['pass3_time_s']['min']:.3f}s | {s['pass3_time_s']['max']:.3f}s | {s['pass3_time_s']['stdev']:.3f}s |",
            f"| **Total**                | {s['total_time_s']['mean']:.3f}s | {s['total_time_s']['median']:.3f}s | {s['total_time_s']['min']:.3f}s | {s['total_time_s']['max']:.3f}s | {s['total_time_s']['stdev']:.3f}s |",
            "",
        ]
    lines += ["---", ""]

lines += [
    "## Raw per-run data",
    "",
    "See `benchmark_log.json` for complete per-run timing of every pass in every scene.",
    "",
]

with open(MD_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n{'='*60}")
print(f"Benchmark complete.")
print(f"  JSON → {LOG_FILE}")
print(f"  MD   → {MD_FILE}")
print(f"{'='*60}")