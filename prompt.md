## ─── PHASE 01 — Project Scaffolding & Environment Setup ───

```
You are a senior full-stack engineer. Your task is to scaffold the complete 
monorepo for a project called "Prompt-to-Blueprint AI" — a system that converts 
natural language descriptions into 2D architectural floor plans.

Create the following directory structure:

prompt-to-blueprint/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── generate.py   # POST /api/v1/generate
│   │   │   │   ├── vastu.py      # POST /api/v1/vastu
│   │   │   │   └── export.py     # POST /api/v1/export/dxf and /svg
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic settings (env vars)
│   │   │   ├── model_registry.py # VRAM manager singleton
│   │   │   └── job_queue.py      # Redis + RQ setup
│   │   ├── models/
│   │   │   ├── schemas.py        # All Pydantic v2 input/output schemas
│   │   │   └── floor_plan.py     # FloorPlan, RoomSpec, AdjacencyEdge models
│   │   ├── services/
│   │   │   ├── nlp_parser.py     # Ollama SLM call + repair logic
│   │   │   ├── gnn_engine.py     # GNN inference wrapper
│   │   │   ├── constraint_solver.py  # Z3 + Shapely pipeline
│   │   │   ├── vastu_engine.py   # Vastu rule checker
│   │   │   └── renderer.py       # SVG + DXF export
│   │   └── workers/
│   │       └── layout_worker.py  # RQ job: chains all services
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── PromptPanel.jsx
│   │   │   ├── CanvasEditor.jsx
│   │   │   ├── ThreePreview.jsx
│   │   │   ├── VastuPanel.jsx
│   │   │   └── ExportToolbar.jsx
│   │   ├── hooks/
│   │   │   └── useJobStream.js   # WebSocket hook
│   │   └── store/
│   │       └── layoutStore.js    # Zustand store
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md

Rules:
- Backend: Python 3.11, FastAPI 0.111, Pydantic v2
- Frontend: React 18, Vite 5, TailwindCSS 3, Konva.js, Zustand
- Do NOT write any ML model code yet — only stubs with TODO comments
- Write real, working boilerplate for FastAPI, React, and Docker Compose
- docker-compose.yml must define 5 services: frontend, api, worker, redis, ollama
- requirements.txt must include: fastapi, uvicorn, pydantic, rq, redis, 
  torch, torch-geometric, shapely, z3-solver, ezdxf, httpx, ollama
- All environment variables go in a .env.example file
- Write a clear README.md with setup instructions

After creating all files, run: find . -type f | sort
to confirm the full tree was created correctly.
```

---

## ─── PHASE 02 — Pydantic Schemas & Data Contracts ───

```
You are a Python backend engineer specialising in data modelling. 

Open backend/app/models/schemas.py and backend/app/models/floor_plan.py.

Your task is to write ALL Pydantic v2 schemas for the Prompt-to-Blueprint AI 
system. These schemas are the single source of truth shared between every 
module — NLP parser output, GNN input/output, API request/response, and 
the frontend JSON contract.

Write the following models with full field validation, type annotations, 
and docstrings:

1. RoomType (Enum)
   Values: LIVING_ROOM, MASTER_BEDROOM, BEDROOM, KITCHEN, BATHROOM, 
   TOILET, CORRIDOR, BALCONY, STUDY, DINING, UTILITY, GARAGE

2. CompassFacing (Enum)  
   Values: NORTH, SOUTH, EAST, WEST

3. ConnectionType (Enum)  
   Values: DOOR, OPENING, WALL

4. RoomSpec (BaseModel)
   Fields: room_id (str), room_type (RoomType), target_area_sqm (float, 
   min=4.0, max=80.0), compass_preference (Optional[CompassFacing]), 
   label (Optional[str])

5. AdjacencyEdge (BaseModel)
   Fields: room_a_id (str), room_b_id (str), connection_type (ConnectionType),
   required (bool, default=True)

6. ParsedLayout (BaseModel)
   Fields: rooms (list[RoomSpec], min_length=2, max_length=12), 
   plot_area_sqm (float, min=30.0, max=500.0), facing (CompassFacing), 
   style_hints (list[str], default=[]), 
   adjacency_constraints (list[AdjacencyEdge]),
   vastu_enabled (bool, default=False)

7. BoundingBox (BaseModel)
   Fields: x_min, y_min, x_max, y_max (all float, normalised 0.0–1.0)
   Add a property: area → float (returns (x_max-x_min)*(y_max-y_min))
   Add a method: to_pixel_coords(plot_width_px, plot_height_px) → tuple[int,int,int,int]

8. RoomLayout (BaseModel)
   Fields: room_spec (RoomSpec), bbox (BoundingBox), 
   door_midpoints (list[tuple[float,float]], default=[])

9. LayoutGraph (BaseModel)
   Fields: rooms (list[RoomLayout]), 
   adjacency_edges (list[AdjacencyEdge]),
   plot_area_sqm (float), facing (CompassFacing),
   generation_mode (Literal["gnn", "heuristic"], default="gnn"),
   overlap_rate (float, default=0.0),
   adjacency_satisfaction (float, default=0.0)

10. GenerateRequest (BaseModel)
    Fields: prompt (str, min_length=10, max_length=800),
    plot_sqm (float, default=100.0), facing (CompassFacing, default=NORTH),
    vastu_enabled (bool, default=False)

11. GenerateResponse (BaseModel)
    Fields: job_id (str), ws_url (str), estimated_seconds (int)

12. JobStatusResponse (BaseModel)
    Fields: status (Literal["queued","parsing","layout_generating",
    "solving_constraints","rendering","complete","error"]),
    progress_pct (int, 0–100), result (Optional[LayoutGraph]),
    svg_string (Optional[str]), error (Optional[ErrorDetail])

13. ErrorDetail (BaseModel)
    Fields: error_code (str), message (str), fallback_used (bool, default=False),
    details (dict, default={})

14. VastuRequest / VastuResponse
    VastuRequest: layout_graph (LayoutGraph)
    VastuResponse: score (int, 0–100), suggestions (list[str]), 
    zone_map (dict[str, str])

15. ExportRequest (BaseModel)
    Fields: layout_graph (LayoutGraph), 
    scale (Literal["1:50","1:100","1:200"], default="1:100")

After writing the models, write 5 unit tests in backend/tests/test_schemas.py 
covering: valid construction, invalid area range, bbox area property, 
to_pixel_coords conversion, and ErrorDetail serialisation.
Run the tests and confirm all 5 pass.
```

---

## ─── PHASE 03 — NLP Parser Service (SLM via Ollama) ───

```
You are an ML engineer building a natural language to structured JSON parser 
using a local Small Language Model served via Ollama.

Open backend/app/services/nlp_parser.py

Your task is to implement the full NLP parser service with these requirements:

FUNCTION 1: build_system_prompt() -> str
  Returns a detailed system prompt that instructs the SLM to:
  - Extract room types, counts, adjacency requirements, plot size, and compass facing
  - Output ONLY valid JSON matching the ParsedLayout schema (imported from models)
  - Never include markdown, explanation, or prose outside the JSON object
  - Handle Indian architectural terminology: BHK, Pooja room (map to STUDY), 
    Verandah (map to BALCONY), Drawing room (map to LIVING_ROOM)
  - Example: "3BHK" = 1 LIVING_ROOM + 3 BEDROOMs (1 MASTER_BEDROOM + 2 BEDROOM)
    + 2 BATHROOMs + 1 KITCHEN

FUNCTION 2: call_ollama(prompt: str, system: str, model: str = "qwen2.5:1.5b") 
            -> dict
  Uses the httpx library to POST to http://localhost:11434/api/chat
  with stream=False. Parses the JSON response and returns the content dict.
  Raises a custom OllamaConnectionError if the server is unreachable.
  Timeout: 30 seconds.

FUNCTION 3: validate_and_repair(raw_json: dict, original_prompt: str, 
            stated_bedroom_count: int) -> ParsedLayout
  - Attempts to construct a ParsedLayout from raw_json
  - If bedroom count in parsed output < stated_bedroom_count, runs a 
    REPAIR PASS: re-calls call_ollama with a chain-of-thought prompt 
    that explicitly lists missing rooms
  - Maximum 2 repair iterations
  - If still invalid after 2 repairs, raises ParseValidationError with details
  - Returns validated ParsedLayout

FUNCTION 4: parse_prompt(user_prompt: str) -> ParsedLayout  (main entrypoint)
  - Extracts bedroom count hint from prompt using a simple regex 
    (look for digits before BHK/bhk/bedroom/Bedroom)
  - Calls call_ollama → validate_and_repair
  - Logs: model used, repair iterations needed, final room count
  - Returns ParsedLayout

FUNCTION 5: async version: aparse_prompt(user_prompt: str) -> ParsedLayout
  Async wrapper using httpx.AsyncClient for use inside FastAPI async routes.

ERROR CLASSES to define in this file:
  - OllamaConnectionError(Exception)
  - ParseValidationError(Exception) — include field: details: dict

Also write integration test stubs in backend/tests/test_nlp_parser.py:
  - test_parse_3bhk_prompt (mock Ollama response)
  - test_repair_pass_triggered (mock under-counted response then correct)
  - test_ollama_unreachable (mock connection error)
Use pytest-mock. Do not require a live Ollama server for tests.
```

---

## ─── PHASE 04 — GNN Layout Engine ───

```
You are a deep learning engineer specialising in Graph Neural Networks for 
spatial layout generation.

Open backend/app/services/gnn_engine.py

Build the complete GNN module for room layout generation using PyTorch Geometric.

PART A — Model Definition: class FloorPlanGNN(torch.nn.Module)

Architecture:
  - Input: node features (18-dim), edge features (3-dim)
  - 3 x GATv2Conv layers (in: 18→128→128→128), 4 attention heads each,
    concat=True on layers 1-2, concat=False on layer 3 (output 128-dim)
  - Edge feature integration: EdgeConv after each GATv2 block
  - Output head: Linear(128, 4) → per-node bounding box [x_min,y_min,x_max,y_max]
  - Output activation: Sigmoid (to keep coords in [0,1])
  - Door head: Linear(128, 2) per edge → door midpoint [x, y]

Node feature vector (18-dim) construction function: build_node_features(room_spec: RoomSpec) -> torch.Tensor
  [0:12]  room_type one-hot (12 classes matching RoomType enum)
  [12]    target_area_sqm normalised (divide by 80.0)
  [13]    adjacency_rank (count of required edges / 6.0)
  [14:18] compass_preference one-hot (N/S/E/W, zeros if None)

Edge feature vector (3-dim): build_edge_features(edge: AdjacencyEdge) -> torch.Tensor
  [0:2]   connection_type one-hot (DOOR, OPENING — WALL maps to [0,0])
  [2]     required flag (1.0 if required else 0.5)

PART B — Layout Graph to PyG Data conversion:
Function: parsed_layout_to_pyg(layout: ParsedLayout) -> torch_geometric.data.Data
  Builds a PyG Data object from a ParsedLayout.
  Nodes = rooms, edges = adjacency_constraints (bidirectional).
  Returns Data(x, edge_index, edge_attr)

PART C — Inference wrapper:
Function: run_gnn_inference(parsed_layout: ParsedLayout, 
          model_path: str = "models/floorplan_gnn.pt") -> LayoutGraph

  1. Loads model weights from model_path (if file doesn't exist, 
     initialise with random weights and log a WARNING — do not crash)
  2. Uses ModelRegistry context manager (from core/model_registry.py) 
     to load model to CUDA if available, else CPU
  3. Converts ParsedLayout → PyG Data → model forward pass
  4. Post-processes raw bbox tensors:
     - Clamps to [0,1]
     - Ensures x_min < x_max and y_min < y_max (swap if violated)
     - Scales to plot dimensions (preserve aspect ratio, assume 10m x 10m default)
  5. Returns LayoutGraph with generation_mode="gnn"

PART D — Training Loss Functions (for future training pipeline):
Function: compute_loss(pred_bboxes, gt_bboxes, pred_edges, gt_adjacency) -> dict
  Returns: {"total": tensor, "giou": tensor, "overlap": tensor, "adj_bce": tensor}
  Weights: giou=1.0, overlap=2.0, adj_bce=0.5
  
  overlap_loss: sum of pairwise intersection areas over all room pairs
  giou_loss: Generalised IoU between predicted and ground-truth bounding boxes
  adj_bce: Binary cross-entropy on whether predicted rooms share a wall

Write unit tests in backend/tests/test_gnn_engine.py:
  - test_node_feature_shape: assert output shape == (18,)
  - test_model_forward_pass: random Data object → assert output shape == (N, 4)
  - test_bbox_clamping: verify no coord outside [0,1] after post-processing
  - test_loss_computation: assert all loss terms are positive scalars
```

---

## ─── PHASE 05 — Constraint Solver & Vastu Engine ───

```
You are a computational geometry engineer. 

Open backend/app/services/constraint_solver.py and vastu_engine.py.

TASK A — Constraint Solver (constraint_solver.py)

Implement a two-stage pipeline that takes a LayoutGraph (which may have 
overlapping or invalid room placements) and returns a physically valid 
LayoutGraph.

STAGE 1 — Z3 Hard Constraint Solver:
Function: solve_with_z3(layout: LayoutGraph, 
          plot_width: float = 10.0, plot_height: float = 10.0,
          timeout_ms: int = 4000) -> tuple[LayoutGraph, bool]

  Returns (solved_layout, success_bool).
  
  Encode these constraints in Z3 (use z3.Real variables for room coordinates):
  1. All coords in [0, plot_width] / [0, plot_height]
  2. For each room: (x_max - x_min) >= 2.4 and (y_max - y_min) >= 2.4 
     (BIS SP 7:2016 minimum dimension)
  3. For each room: area = (x_max-x_min)*(y_max-y_min) within 20% of target
  4. For each pair of rooms: no overlap 
     (encode as: x1_max<=x2_min OR x2_max<=x1_min OR y1_max<=y2_min OR y2_max<=y1_min)
  5. For each required adjacency edge: shared wall length >= 0.9m
     (rooms must touch on at least one axis within 0.9m)
  6. All rooms contained within plot boundary
  
  Use z3.set_option("timeout", timeout_ms)
  If sat: extract model values, rebuild LayoutGraph, return (layout, True)
  If unsat or timeout: return (original_layout, False)

STAGE 2 — Soft Optimisation with Shapely + SciPy:
Function: optimise_layout(layout: LayoutGraph, 
          plot_width: float, plot_height: float) -> LayoutGraph

  Uses scipy.optimize.minimize with L-BFGS-B to adjust room positions 
  maximising:
  - Living room and master bedroom proximity to east/south edges (natural light)
  - Minimise total corridor area (target < 8% of gross floor area)
  - Kitchen-to-bedroom centroid distance >= 3.0m
  
  The objective function uses Shapely Polygon objects for area calculations.
  Constraint: do not re-introduce overlaps (add overlap penalty to objective).

STAGE 3 — Heuristic Fallback Placer:
Function: heuristic_place(layout: LayoutGraph, 
          plot_width: float, plot_height: float) -> LayoutGraph

  Strip-packing algorithm. Place rooms in this fixed priority order:
  LIVING_ROOM → MASTER_BEDROOM → KITCHEN → BATHROOM/TOILET → 
  BEDROOM → STUDY → DINING → CORRIDOR → BALCONY → UTILITY → GARAGE
  
  Each room gets a strip of width=plot_width/2 or plot_width, stacked 
  vertically. Rooms within a strip are placed left-to-right.
  Mark layout.generation_mode = "heuristic"

MAIN ENTRYPOINT:
Function: run_constraint_pipeline(layout: LayoutGraph, 
          plot_width: float = 10.0, 
          plot_height: float = 10.0) -> LayoutGraph

  1. Run solve_with_z3 → if success, run optimise_layout → return
  2. If Z3 fails → run heuristic_place → run optimise_layout → return
  3. Log which path was taken
  4. Calculate and set layout.overlap_rate and layout.adjacency_satisfaction

TASK B — Vastu Engine (vastu_engine.py)

Build a 9-zone Vastu grid checker.

Function: get_vastu_zone(x_norm: float, y_norm: float) -> str
  Divides plot into 3x3 = 9 zones (NW, N, NE, W, CENTER, E, SW, S, SE)
  based on normalised [0,1] coordinates.

VASTU_RULES dict: maps (zone, RoomType) → score (0–100)
  Best placements (score 100): KITCHEN→SE, MASTER_BEDROOM→SW, 
  LIVING_ROOM→NE or N, BATHROOM→NW or W, STUDY→NE, POOJA→NE
  Neutral (score 60): any room in CENTER
  Bad placement (score 20): KITCHEN→NE, MASTER_BEDROOM→NE or SE

Function: check_vastu(layout: LayoutGraph) -> VastuResponse
  For each room, find its zone (use centroid of BoundingBox).
  Look up score from VASTU_RULES (default 60 if not found).
  Compute composite score = weighted average (weight by room area).
  Generate suggestions list for any room scoring below 50.
  Return VastuResponse(score, suggestions, zone_map)

Write tests for both modules covering: overlap detection, 
minimum dimension enforcement, zone assignment at boundaries, 
and vastu score computation for a known layout.
```

---

## ─── PHASE 06 — Renderer (SVG + DXF Export) ───

```
You are a technical graphics engineer. 

Open backend/app/services/renderer.py

Implement the rendering service that converts a LayoutGraph into visual outputs.

FUNCTION 1: layout_to_svg(layout: LayoutGraph, 
            canvas_width_px: int = 800, 
            canvas_height_px: int = 800,
            show_dimensions: bool = True,
            show_room_labels: bool = True) -> str

  Build SVG using Python's xml.etree.ElementTree (no external SVG library).
  
  SVG structure:
  - Root <svg> with viewBox, xmlns attributes
  - <defs> block with room-type colour palette:
      LIVING_ROOM: #E8F4FD, KITCHEN: #FFF9E6, MASTER_BEDROOM: #F0F4FF,
      BEDROOM: #F5F0FF, BATHROOM: #E6F9F5, CORRIDOR: #F5F5F5,
      BALCONY: #E8F8E8, STUDY: #FFF0E8, default: #FAFAFA
  - For each room: <rect> with x,y,width,height (converted from normalised 
    BoundingBox using to_pixel_coords), fill from palette, 
    stroke="#334155", stroke-width="2", rx="2"
    data-room-type and data-room-id attributes for frontend interactivity
  - Room label: <text> centred in room, font-family="Arial", font-size="11",
    fill="#1E293B", content = room_type display name + area in m²
  - Dimension lines: small <line> + <text> elements showing room width and 
    height in metres along the room edges (only if show_dimensions=True)
  - Door markers: small arc <path> at each door_midpoint
  - Plot border: <rect> around full canvas, stroke="#0D1F3C", stroke-width="3"
  
  Return the complete SVG as a UTF-8 string.

FUNCTION 2: layout_to_dxf(layout: LayoutGraph, 
            plot_width_m: float = 10.0,
            plot_height_m: float = 10.0,
            scale_str: str = "1:100") -> bytes

  Use ezdxf to build a DXF R2018 document.
  
  Layers to create:
  - WALLS (color=7, lineweight=50)
  - DOORS (color=3, lineweight=25)
  - DIMENSIONS (color=2, lineweight=13)
  - ANNOTATIONS (color=1, lineweight=13)
  - PLOT_BOUNDARY (color=5, lineweight=70)
  
  For each room:
  - Draw 4 wall lines on WALLS layer using real-world metres
    (multiply normalised coords by plot_width_m / plot_height_m)
  - Draw MTEXT annotation on ANNOTATIONS layer at room centroid:
    "{RoomType} — {area:.1f}m²"
  - Draw door arc on DOORS layer at each door_midpoint
  
  Add DIMLINEAR dimensions along bottom and left edges of each room 
  on DIMENSIONS layer.
  
  Apply scale factor from scale_str (e.g. 1:100 → multiply all 
  coordinates by 100 for paper space).
  
  Return dxf.write_zip() bytes.

FUNCTION 3: compute_overlap_rate(layout: LayoutGraph) -> float
  Uses Shapely to compute total overlapping area / total floor area.
  Returns float between 0.0 and 1.0.

FUNCTION 4: compute_adjacency_satisfaction(layout: LayoutGraph) -> float
  Checks each required AdjacencyEdge in the layout.
  Two rooms are "adjacent" if their Shapely polygons share a boundary 
  of length >= 0.9m (in normalised units scaled to 10m plot = 0.09).
  Returns fraction of required edges that are satisfied.

Write tests in backend/tests/test_renderer.py:
  - test_svg_output_is_valid_xml
  - test_svg_contains_all_rooms
  - test_dxf_layers_created
  - test_overlap_rate_zero_for_non_overlapping
  - test_overlap_rate_positive_for_overlapping
```

---

## ─── PHASE 07 — Model Registry & Job Queue ───

```
You are a systems engineer focused on resource management and async processing.

TASK A — Open backend/app/core/model_registry.py

Implement a thread-safe singleton ModelRegistry that manages VRAM loading 
and eviction of ML models.

class ModelRegistry:
  - _instance: class-level singleton variable
  - _current_model_name: str | None
  - _lock: threading.Lock
  
  Method: load(model_name: str, loader_fn: Callable) -> Any
    Context manager pattern:
    1. Acquire lock
    2. If _current_model_name != model_name:
       - Call self.evict() first
       - Call loader_fn() to load new model
       - Set _current_model_name = model_name
    3. Yield the loaded model
    4. Do NOT evict on exit (keep model warm for repeated calls)
  
  Method: evict()
    - Sets model reference to None
    - Calls torch.cuda.empty_cache() if CUDA available
    - Logs: "Evicted {model_name} from VRAM, freeing {freed_mb:.0f}MB"
    - Updates _current_model_name = None
  
  Method: status() -> dict
    Returns: { "current_model": str|None, "vram_used_gb": float, 
               "vram_free_gb": float, "ram_used_gb": float }
  
  Implement __new__ for singleton pattern.
  Write a context manager: registry.use(model_name, loader_fn) 
  that auto-evicts the PREVIOUS model before loading the new one.

TASK B — Open backend/app/core/job_queue.py

Implement the async job management layer using Redis + RQ.

Functions to implement:

enqueue_layout_job(request: GenerateRequest) -> str
  - Generates a unique job_id = sha256(prompt + str(plot_sqm) + facing)[:16]
  - Checks Redis cache: if job_id exists with status "complete", 
    return job_id immediately (cache hit)
  - Otherwise enqueues layout_worker.run_layout_job to the RQ "layout" queue
  - Sets Redis key job:{job_id}:status = "queued" with TTL 86400 (24h)
  - Returns job_id

get_job_status(job_id: str) -> JobStatusResponse
  - Reads job:{job_id}:status from Redis
  - If "complete", also reads job:{job_id}:result (JSON) and 
    job:{job_id}:svg from Redis
  - Returns JobStatusResponse

update_job_progress(job_id: str, status: str, progress_pct: int, 
                    result: dict | None = None)
  - Writes to Redis with TTL 86400
  - Publishes to Redis channel "job:{job_id}:events" for WebSocket relay

TASK C — Open backend/workers/layout_worker.py

Implement the RQ worker function that chains all services:

def run_layout_job(job_id: str, request_dict: dict):
  request = GenerateRequest(**request_dict)
  
  Step 1: update_job_progress(job_id, "parsing", 10)
          parsed = parse_prompt(request.prompt)
  
  Step 2: update_job_progress(job_id, "layout_generating", 30)
          raw_layout = run_gnn_inference(parsed)
  
  Step 3: update_job_progress(job_id, "solving_constraints", 55)
          solved_layout = run_constraint_pipeline(raw_layout)
          if request.vastu_enabled:
            vastu = check_vastu(solved_layout)
          else:
            vastu = None
  
  Step 4: update_job_progress(job_id, "rendering", 80)
          svg = layout_to_svg(solved_layout)
          solved_layout.overlap_rate = compute_overlap_rate(solved_layout)
          solved_layout.adjacency_satisfaction = compute_adjacency_satisfaction(solved_layout)
  
  Step 5: update_job_progress(job_id, "complete", 100, 
          result=solved_layout.model_dump())
          Store svg in Redis: job:{job_id}:svg

  Wrap entire function in try/except. On exception:
  update_job_progress(job_id, "error", 0, 
    result=ErrorDetail(error_code="WORKER_EXCEPTION", 
    message=str(e)).model_dump())

Write tests for ModelRegistry: test_singleton_pattern, test_evict_clears_state.
Write tests for job queue: test_cache_hit_returns_same_job_id, 
test_progress_update_writes_redis (use fakeredis).
```

---

## ─── PHASE 08 — FastAPI Routes & WebSocket ───

```
You are a backend API engineer. 

Open backend/app/main.py and all files in backend/app/api/routes/.

TASK A — main.py

Set up the FastAPI application with:
  - CORS middleware (allow origins from env var FRONTEND_URL, default localhost:3000)
  - Lifespan handler: on startup verify Redis connection and Ollama reachability,
    log warnings (do NOT crash) if either is unavailable
  - Include routers: generate, vastu, export, health
  - Mount static files for any SVG previews at /static
  - Custom exception handlers for OllamaConnectionError → HTTP 503,
    ParseValidationError → HTTP 422, generic Exception → HTTP 500
  - All errors return ErrorDetail JSON body

TASK B — routes/generate.py

POST /api/v1/generate
  - Accepts GenerateRequest
  - Calls enqueue_layout_job(request) → job_id
  - Returns GenerateResponse(job_id, ws_url=f"ws://localhost:8000/ws/{job_id}", 
    estimated_seconds=8)

GET /api/v1/jobs/{job_id}
  - Returns JobStatusResponse from get_job_status(job_id)
  - Returns 404 if job_id not found in Redis

WebSocket endpoint: /ws/{job_id}
  - On connect: immediately send current job status
  - Subscribe to Redis pubsub channel "job:{job_id}:events" 
  - Relay all published messages to the WebSocket client as JSON
  - On disconnect: unsubscribe and clean up
  - Ping client every 15 seconds to keep connection alive
  - Use asyncio with redis.asyncio (aioredis pattern)

GET /api/v1/health
  - Returns: { "status": "ok", "redis": bool, "ollama": bool,
    "vram_free_gb": float, "current_model": str|None }
  - Calls ModelRegistry.status() for VRAM info

TASK C — routes/vastu.py
  POST /api/v1/vastu
    Accepts VastuRequest, calls check_vastu(request.layout_graph)
    Returns VastuResponse. Must complete in < 500ms.

TASK D — routes/export.py
  POST /api/v1/export/dxf
    Accepts ExportRequest. Returns FileResponse with 
    Content-Disposition: attachment; filename="floor_plan.dxf"
    media_type: "application/dxf"
  
  POST /api/v1/export/svg
    Accepts ExportRequest (only uses layout_graph).
    Returns { "svg_string": str }

Run the FastAPI app and confirm all routes appear in /docs (Swagger UI).
Fix any import errors before proceeding to the frontend phase.
```

---

## ─── PHASE 09 — React Frontend ───

```
You are a senior React engineer. Build the complete frontend for 
Prompt-to-Blueprint AI.

TASK A — frontend/src/store/layoutStore.js
Zustand store with these slices:
  - prompt: string
  - jobId: string | null
  - jobStatus: string
  - progressPct: number
  - layoutGraph: object | null
  - svgString: string | null
  - vastuResult: object | null
  - history: array (max 50 snapshots of layoutGraph)
  - historyIndex: number
  Actions: setPrompt, setJobId, setJobStatus, setLayoutGraph, setSvgString,
           setVastuResult, pushHistory, undo, redo, reset

TASK B — frontend/src/hooks/useJobStream.js
Custom hook: useJobStream(jobId)
  - On mount (when jobId is not null): open WebSocket to ws://localhost:8000/ws/{jobId}
  - Parse incoming JSON messages, dispatch to Zustand store
  - On message type "complete": set layoutGraph, svgString, progressPct=100
  - On message type "error": set error state
  - On unmount: close WebSocket
  - Reconnect with exponential backoff (max 3 retries) on disconnect
  - Return: { connected: bool, lastMessage: object | null }

TASK C — frontend/src/components/PromptPanel.jsx
  - Large textarea for prompt input (min 3 rows, max 10)
  - Sidebar controls (collapsible):
      • Plot size slider: 40–300 m² 
      • Compass facing: 4 toggle buttons (N/S/E/W) with compass icon
      • Vastu toggle switch
  - "Generate Floor Plan" button (disabled while job running)
  - Progress bar: 5-segment bar showing PARSING → GENERATING → SOLVING → 
    RENDERING → COMPLETE, each segment lights up as WebSocket events arrive
  - Error display: red banner with error message and "Try Again" button
  - Calls POST /api/v1/generate on submit, stores job_id in Zustand

TASK D — frontend/src/components/CanvasEditor.jsx
  - Uses Konva.js Stage and Layer
  - When layoutGraph arrives: render each room as KonvaRect
    - Fill with room-type colour palette (match backend SVG colours)
    - Stroke: #334155, strokeWidth: 2
    - Room label: KonvaText centred in rect showing room type + area
    - Doors: small KonvaArc at door midpoints
  - Drag behaviour: rooms are draggable
    - On drag-end: update room bbox in Zustand store, push to history,
      POST updated layout to /api/v1/vastu to refresh Vastu score
  - Zoom: mouse wheel zoom (0.5x–3x range)
  - Pan: middle-mouse drag to pan canvas
  - Selection: click room to show resize handles (use KonvaTransformer)

TASK E — frontend/src/components/VastuPanel.jsx
  - Displays VastuResponse from store
  - Score shown as animated circular progress gauge (SVG-based, no library)
  - Score colour: red (<40), amber (40–70), green (>70)
  - List of suggestions with directional arrow icons
  - 3x3 Vastu zone grid visual showing each room in its zone with colour coding

TASK F — frontend/src/components/ExportToolbar.jsx
  - "Download DXF" button: POST to /api/v1/export/dxf, trigger file download
  - "Copy SVG" button: copies svg_string from store to clipboard
  - "Undo" / "Redo" buttons with keyboard shortcuts (Ctrl+Z, Ctrl+Y)
  - Scale selector: 1:50 / 1:100 / 1:200

TASK G — frontend/src/components/ThreePreview.jsx (optional toggle)
  - Hidden by default, shown via "3D Preview" toggle button
  - Uses @react-three/fiber + @react-three/drei
  - For each room in layoutGraph: create a BoxGeometry extruded to height 2.8m
  - Different MeshStandardMaterial colour per room type
  - OrbitControls for rotation/zoom
  - Ambient + directional lighting

TASK H — frontend/src/App.jsx
  Main layout:
  - Left panel (30% width): PromptPanel
  - Centre panel (50% width): CanvasEditor (tabs: 2D Plan | 3D Preview)
  - Right panel (20% width): VastuPanel
  - Bottom bar: ExportToolbar + health status indicator (green/red dot)
  Use TailwindCSS grid for layout. Responsive: stack vertically on mobile.

Run: npm run dev and fix all console errors before the next phase.
```

---

## ─── PHASE 10 — Training Pipeline (GNN Fine-tuning) ───

```
You are an ML engineer. Build the GNN training pipeline.
Create a new directory: backend/training/

Files to create:

1. training/dataset.py
   Class: FloorPlanDataset(torch_geometric.data.Dataset)
   - __init__(root, split="train") where split ∈ {train, val, test}
   - Looks for processed graph files in root/processed/{split}/
   - processed() method: reads raw JSON files from root/raw/, 
     converts each ParsedLayout JSON + LayoutGraph JSON pair using 
     parsed_layout_to_pyg() from gnn_engine.py
   - Each item: Data(x=node_feats, edge_index, edge_attr, 
                     y_bbox=ground_truth_bboxes)
   - Train/val/test split: 80/10/10 by file index
   - __len__ and __getitem__ implementations

2. training/train_gnn.py
   Full training script with these sections:
   
   CONFIG (argparse):
     --data_root, --epochs (default 100), --lr (default 1e-3),
     --batch_size (default 16), --device (default "cuda" if available),
     --checkpoint_dir (default "models/"), --wandb (flag, default False)
   
   TRAINING LOOP:
     - DataLoader with batch_size, shuffle=True for train
     - AdamW optimiser, ReduceLROnPlateau scheduler (patience=10)
     - Per-epoch: forward pass → compute_loss → backprop → scheduler step
     - Log: epoch, total_loss, giou_loss, overlap_loss, adj_bce_loss
     - Validation every 5 epochs: compute mean IoU on val set
     - Save best checkpoint (lowest val loss) to checkpoint_dir/best_gnn.pt
     - Early stopping: stop if val loss does not improve for 20 epochs
   
   METRICS FUNCTION: evaluate_model(model, loader) -> dict
     Returns: {mean_iou, overlap_rate, adjacency_satisfaction}
     Uses compute_overlap_rate and compute_adjacency_satisfaction from renderer.py

3. training/synthetic_data_generator.py
   Script to generate the synthetic training corpus.
   
   Function: generate_prompts_for_graph(layout_json: dict, 
             model: str = "mistral:8b") -> list[str]
     Calls Ollama with a reverse-engineering prompt:
     "Given this floor plan graph: {layout_json}
      Write 3 different natural language descriptions a homeowner might use 
      to request this floor plan. Vary between casual, formal, and terse styles.
      Return a JSON array of 3 strings only."
     Returns list of 3 prompt strings.
   
   Function: validate_prompt_pair(prompt: str, layout: dict, 
             min_f1: float = 0.85) -> bool
     Runs the NLP parser on prompt, computes room-type F1 against layout.
     Returns True if F1 >= min_f1.
   
   Main script logic:
   - Reads all CubiCasa5k graph JSON files from data/raw/cubicasa5k/
   - For each graph: generate 3 prompts → validate each → save valid pairs
   - Progress bar (tqdm), parallel processing (ThreadPoolExecutor, max_workers=4)
   - Output: data/synthetic/corpus.jsonl (one {"prompt": str, "graph": dict} per line)
   - Log: total pairs generated, pairs rejected, acceptance rate

4. training/evaluate.py
   Evaluation script that produces a metrics report:
   - Loads best checkpoint
   - Runs on test set
   - Prints table: Mean IoU | Overlap Rate | Adj. Satisfaction | Solve Rate
   - Saves report to models/eval_report.json

Add a Makefile at project root with targets:
  make train       → runs train_gnn.py with default args
  make generate-data → runs synthetic_data_generator.py
  make evaluate    → runs evaluate.py
  make serve       → starts docker-compose up -d
```

---

## ─── PHASE 11 — Integration Testing & E2E Validation ───

```
You are a QA and DevOps engineer. Your task is to write a complete 
integration and E2E test suite for Prompt-to-Blueprint AI.

Create the following test files:

1. backend/tests/test_integration_pipeline.py
   Full pipeline integration tests (use pytest + pytest-asyncio):
   
   Fixture: mock_parsed_layout() → returns a hardcoded ParsedLayout 
   representing a 2BHK flat (living room, 2 bedrooms, kitchen, 2 bathrooms)
   
   test_full_pipeline_2bhk:
     - Run mock_parsed_layout through run_gnn_inference (random weights OK)
     - Run output through run_constraint_pipeline
     - Assert: overlap_rate < 0.01 (Z3 or heuristic must eliminate overlaps)
     - Assert: all rooms have min dimension >= 2.4m (in 10x10m plot)
     - Assert: generation_mode in ["gnn", "heuristic"]
   
   test_renderer_svg_valid:
     - Run above pipeline, pass LayoutGraph to layout_to_svg
     - Parse the SVG string with xml.etree.ElementTree
     - Assert: number of <rect> elements == number of rooms
     - Assert: all data-room-type attributes are valid RoomType values
   
   test_renderer_dxf_valid:
     - Run layout_to_dxf on mock layout
     - Load the bytes with ezdxf.read_zip()
     - Assert: layers WALLS, DOORS, DIMENSIONS, ANNOTATIONS all exist
   
   test_vastu_score_range:
     - Run check_vastu on mock layout
     - Assert: 0 <= score <= 100
     - Assert: isinstance(suggestions, list)
   
   test_api_generate_endpoint (use FastAPI TestClient):
     - POST /api/v1/generate with a sample prompt
     - Assert: 200 status, response has job_id and ws_url
   
   test_api_health_endpoint:
     - GET /api/v1/health
     - Assert: 200 status, response has "status" key

2. frontend/src/__tests__/PromptPanel.test.jsx (using Vitest + React Testing Library)
   - test: renders textarea and generate button
   - test: button disabled when jobStatus is "parsing"
   - test: calls onSubmit with correct payload on button click

3. frontend/src/__tests__/layoutStore.test.js
   - test: undo/redo history works correctly
   - test: pushHistory trims to 50 items max
   - test: reset clears all state

4. docker-compose.test.yml
   Adds a "test-runner" service that:
   - Depends on api, redis, ollama
   - Runs: pytest backend/tests/ -v --tb=short
   - Exits with test result code

5. .github/workflows/ci.yml (GitHub Actions)
   On push to main and pull requests:
   - Job: backend-tests
       Python 3.11, install requirements, run pytest with coverage
       Coverage threshold: fail if < 70%
   - Job: frontend-tests  
       Node 20, npm ci, run vitest
   - Job: docker-build
       Build both Dockerfile images, assert build succeeds

Run pytest backend/tests/ -v from the project root.
Fix all failing tests. Target: all integration tests passing.
Report final test count and coverage percentage.
```

---

## ─── PHASE 12 — Final Polish, Docs & Deployment ───

```
You are a senior engineer doing final project hardening and documentation.

TASK A — Performance Audit
  Profile the full pipeline end-to-end:
  1. Add Python cProfile decorator to run_layout_job in layout_worker.py
  2. Run the pipeline 3 times with a standard "3BHK with open kitchen" prompt
  3. Report: mean total latency, latency per stage, peak VRAM used
  4. If any stage exceeds these budgets, optimise it:
     - NLP parsing: > 3s → reduce max_tokens in Ollama call to 512
     - GNN inference: > 2s → add torch.inference_mode() decorator
     - Z3 solving: > 4s → already has timeout, ensure fallback triggers correctly
     - SVG rendering: > 0.5s → profile xml.etree calls, cache room colour lookups

TASK B — README.md (overwrite the stub from Phase 1)
  Write a complete README.md with these sections:
  - Project overview (2–3 sentences)
  - Architecture diagram (ASCII art showing the 5-layer pipeline)
  - Hardware requirements (minimum: 6GB VRAM, 16GB RAM, RTX 30/40 series)
  - Quick start: git clone → cp .env.example .env → docker-compose up -d
  - Ollama model setup: ollama pull qwen2.5:1.5b
  - How to train the GNN: make generate-data → make train
  - API reference table (all endpoints)
  - Project structure tree
  - Evaluation metrics table (with target values)
  - Known limitations section
  - Contributing guide

TASK C — Environment & Secrets Hardening
  - Audit all hardcoded values (ports, model names, timeouts) → move to .env
  - Add input sanitisation to the /generate endpoint: 
    strip prompt of HTML/JS, max 800 chars enforced at route level
  - Add rate limiting: max 10 requests/minute per IP using slowapi
  - Add request ID middleware: attach X-Request-ID header to every response

TASK D — docker-compose.yml Final Review
  - Add resource limits to each service:
      api and worker: mem_limit: 8g, cpus: "4"
      redis: mem_limit: 512m
      ollama: mem_limit: 8g, deploy.resources.reservations.devices (GPU)
  - Add healthcheck to api service: 
      test: curl -f http://localhost:8000/api/v1/health || exit 1
      interval: 30s, timeout: 10s, retries: 3
  - Add restart: unless-stopped to all services

TASK E — Final Smoke Test
  Run these commands in order and confirm each succeeds:
  1. docker-compose up -d --build
  2. curl http://localhost:8000/api/v1/health  → expect {"status":"ok"}
  3. curl -X POST http://localhost:8000/api/v1/generate \
       -H "Content-Type: application/json" \
       -d '{"prompt":"Create a 2BHK flat with east facing entrance","plot_sqm":80}'
     → expect {"job_id": "...", "ws_url": "..."}
  4. curl http://localhost:8000/api/v1/jobs/{job_id_from_step_3}
     → poll until status == "complete"
  5. Open http://localhost:3000 in browser → submit the same prompt → 
     confirm floor plan renders on canvas

Report any failures and fix them before declaring the project complete.
```

---
