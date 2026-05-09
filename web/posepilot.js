import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// PosePilot.Viewfinder — favorite views for PosePilot3D
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "PosePilot.Viewfinder",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PosePilot3D") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const node = this;

      function getWidget(name) {
        return node.widgets?.find(w => w.name === name);
      }

      function getGlbPath() {
        return getWidget("glb_path")?.value?.trim() ?? "";
      }

      // Read camera_info from the Load3D widget's serialized value.
      function getCurrentCamera() {
        const w = getWidget("image");
        const v = w?.value;
        if (v && typeof v === "object" && v.camera_info) return v.camera_info;
        return null;
      }

      // Rebuild view_preset combo options from the server.
      async function refreshViews(glbPath) {
        if (!glbPath) return;
        const presetW = getWidget("view_preset");
        if (!presetW) return;
        try {
          const resp = await api.fetchApi(
            `/posepilot/get_views?glb_path=${encodeURIComponent(glbPath)}`
          );
          const data = await resp.json();
          const views = data.views ?? [];
          const options = [
            "(live camera)",
            ...views.map((v, i) => `${i + 1}: ${v.label || "untitled"}`),
          ];
          presetW.options.values = options;
          if (!options.includes(presetW.value)) {
            presetW.value = options[0];
          }
          app.graph.setDirtyCanvas(true);
        } catch (e) {
          console.warn("[PosePilot] get_views failed:", e);
        }
      }

      // Refresh view list whenever glb_path changes (new pose selected).
      const glbW = getWidget("glb_path");
      if (glbW) {
        let _glbVal = glbW.value ?? "";
        Object.defineProperty(glbW, "value", {
          get() { return _glbVal; },
          set(v) { _glbVal = v; refreshViews(v); },
          configurable: true,
        });
      }

      // "Save View" button
      node.addWidget("button", "Save View", null, async () => {
        const glbPath = getGlbPath();
        if (!glbPath) {
          alert("[PosePilot] Connect PosePilotSelector first — glb_path is required to save views.");
          return;
        }
        const camera = getCurrentCamera();
        if (!camera) {
          alert("[PosePilot] No camera data yet. Execute the node once so the viewport captures a camera position.");
          return;
        }
        const label = (getWidget("view_label")?.value ?? "").trim();
        try {
          const resp = await api.fetchApi("/posepilot/save_view", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ glb_path: glbPath, label, camera }),
          });
          const data = await resp.json();
          if (data.ok) {
            await refreshViews(glbPath);
            // Auto-select the view that was just saved.
            const presetW = getWidget("view_preset");
            if (presetW?.options?.values?.length > 1) {
              presetW.value = presetW.options.values[presetW.options.values.length - 1];
            }
            const labelW = getWidget("view_label");
            if (labelW) labelW.value = "";
            app.graph.setDirtyCanvas(true);
          }
        } catch (e) {
          console.warn("[PosePilot] save_view failed:", e);
        }
      });

      // "Update View" button — moves the viewport camera to the selected favorite.
      // Mutates widget.value.camera_info and redraws; whether the Three.js viewport
      // visually updates depends on ComfyUI internals. OpenPose output is always
      // correct on execute regardless (server-side override handles it).
      node.addWidget("button", "Update View (execute to apply)", null, async () => {
        const glbPath = getGlbPath();
        const presetW = getWidget("view_preset");
        if (!glbPath || !presetW || presetW.value === "(live camera)") return;

        const match = presetW.value.match(/^(\d+):/);
        if (!match) return;
        const idx = parseInt(match[1], 10) - 1;  // 0-based

        try {
          const resp = await api.fetchApi(
            `/posepilot/get_views?glb_path=${encodeURIComponent(glbPath)}`
          );
          const data = await resp.json();
          const views = data.views ?? [];
          if (idx < 0 || idx >= views.length) return;

          const savedCamera = views[idx].camera;
          const imageW = getWidget("image");
          if (imageW) {
            const current = (typeof imageW.value === "object" && imageW.value) ? imageW.value : {};
            imageW.value = { ...current, camera_info: savedCamera };
          }
          app.graph.setDirtyCanvas(true);
        } catch (e) {
          console.warn("[PosePilot] update_view failed:", e);
        }
      });

      // "Delete View" button — removes the currently selected favorite.
      node.addWidget("button", "Delete View", null, async () => {
        const glbPath = getGlbPath();
        const presetW = getWidget("view_preset");
        if (!glbPath || !presetW || presetW.value === "(live camera)") return;

        const match = presetW.value.match(/^(\d+):/);
        if (!match) return;
        const idx = parseInt(match[1], 10) - 1;  // 0-based

        try {
          const resp = await api.fetchApi("/posepilot/delete_view", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ glb_path: glbPath, index: idx }),
          });
          const data = await resp.json();
          if (data.ok) {
            presetW.value = "(live camera)";
            await refreshViews(glbPath);
            app.graph.setDirtyCanvas(true);
          }
        } catch (e) {
          console.warn("[PosePilot] delete_view failed:", e);
        }
      });
    };
  },
});

// ---------------------------------------------------------------------------
// PosePilot.Selector — library and pose dropdowns (canonical models/pose-pilot/)
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "PosePilot.Selector",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PosePilotSelector") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const node = this;
      const libW  = node.widgets?.find(w => w.name === "library");
      const poseW = node.widgets?.find(w => w.name === "pose_file");
      if (!libW || !poseW) return;

      async function refreshPoses(library) {
        if (!library?.trim() || library === "") return;
        try {
          const resp = await api.fetchApi(
            `/posepilot/list_glbs?library=${encodeURIComponent(library)}`
          );
          const data = await resp.json();
          const files = data.files ?? [];
          if (files.length === 0) return;
          poseW.options.values = files;
          if (!files.includes(poseW.value)) {
            poseW.value = files[0];
          }
          app.graph.setDirtyCanvas(true);
        } catch (e) {
          console.warn("[PosePilot] Pose list refresh failed:", e);
        }
      }

      async function refreshLibraries() {
        try {
          const resp = await api.fetchApi("/posepilot/list_libraries");
          const data = await resp.json();
          const libs = data.libraries ?? [];
          if (libs.length === 0) return;
          libW.options.values = libs;
          if (!libs.includes(libW.value)) {
            libW.value = libs[0];
          }
          // Poses for the current library may not be loaded yet on first paint.
          await refreshPoses(libW.value);
          app.graph.setDirtyCanvas(true);
        } catch (e) {
          console.warn("[PosePilot] Library list refresh failed:", e);
        }
      }

      // Combo widgets fire widget.callback on user selection — that's the
      // reliable signal for dropdowns (Object.defineProperty on .value works
      // for string inputs but not for combo selection events).
      const _origLibCallback = libW.callback;
      libW.callback = function(value) {
        _origLibCallback?.call(this, value);
        refreshPoses(value);
      };

      // When a saved workflow is loaded, LiteGraph restores widget values via
      // node.configure() AFTER onNodeCreated — hook it to repopulate poses.
      const _origConfigure = node.onConfigure;
      node.onConfigure = function(info) {
        _origConfigure?.call(node, info);
        const lib = libW.value;
        if (lib) refreshPoses(lib);
      };

      // Populate both dropdowns when the node is first created.
      refreshLibraries();
    };
  },
});
