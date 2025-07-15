/* queue_manager.js – handles generation queue + mapper list */

function _defaultDiffLabel(stars) {
    if (stars < 2.5) return "Easy";
    else if (stars < 3.5) return "Normal";
    else if (stars < 4.5) return "Hard";
    else if (stars < 5.5) return "Insane";
    else return "Expert";
}

/* ------------------------------------------------------------- */
/*  QUEUE CORE                                                   */
/* ------------------------------------------------------------- */
const QueueManager = (() => {
    const queue = [];
    let running = false;

    function render() {
        const list = document.getElementById("queue-list");
        list.innerHTML = "";
        queue.forEach((task, idx) => {
            const li = document.createElement("li");
            li.classList.add("queue-item");
            li.textContent =
                `${task.display_name || task.audio_path} → ${task.output_path} ` +
                `(mapper ${task.mapper_display_name || "—"})`;
            const del = document.createElement("button");
            del.textContent = "✕";
            del.className = "delete-btn";
            del.onclick = () => remove(idx);
            li.appendChild(del);
            if (idx === 0 && running) li.classList.add("running");
            list.appendChild(li);
        });
    }

    /* ---------- public helpers ---------- */
    function add(task) { queue.push(task); render(); }
    function clear() { if (!running) { queue.length = 0; render(); } }
    function remove(i) { if (running && i === 0) return; queue.splice(i, 1); render(); }
    function hasPending() { return queue.length > 0; }
    function isRunning() { return running; }

    /* ---------- execution loop ---------- */
    async function _runNext() {
        if (!queue.length) { running = false; render(); return; }

        running = true;
        const task = queue.shift();     // <── key fix: shrink queue *before* submit
        render();

        try {
            await InferenceManager.runTask(task);   // see below
        } catch (e) {
            console.error("Task failed", e);
        } finally {
            _runNext();                 // recurse
        }
    }
    function start() { if (!running && queue.length) _runNext(); }

    return { add, clear, remove, hasPending, isRunning, start, render };
})();


/* ------------------------------------------------------------------ */
/* Mapper list manager */

const MapperManager = (() => {
    const listEl = document.getElementById("mappers-list");

    function _entryTemplate(id, name) {
        return `
      <div class="mapper-item" data-id="${id}">
        <input type="checkbox" class="mapper-check" checked>
        <input type="text" class="mapper-name" value="${name}">
        <input type="number" class="mapper-count" value="1" min="1" class="count-input">
        <button type="button" class="remove-mapper-btn">✕</button>
      </div>`;
    }

    async function addMapper(id) {
        id = id.trim();
        if (!id) return;
        // ask backend for username
        let name = "Loading…";
        try {
            const res = await fetch("/lookup_mapper_name", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mapper_id: id })
            });
            const data = await res.json();
            name = data.username || ("ID " + id);
        } catch (e) { name = "ID " + id; }
        listEl.insertAdjacentHTML("beforeend", _entryTemplate(id, name));
    }
    /* ----------  NEW:  export / import helpers  ---------- */

    function getAll() {
        return Array.from(listEl.querySelectorAll(".mapper-item")).map(div => ({
            id: div.dataset.id,
            name: div.querySelector(".mapper-name").value.trim(),
            n: parseInt(div.querySelector(".mapper-count").value, 10) || 1,
            checked: div.querySelector(".mapper-check").checked
        }));
    }

    function clearAll() {
        listEl.innerHTML = "";
    }

    /** Accepts the array produced by getAll() */
    function loadFromArray(arr) {
        clearAll();
        arr.forEach(obj => {
            const { id, name, n = 1, checked = true } = obj;
            listEl.insertAdjacentHTML("beforeend", _entryTemplate(id, name));
            const div = listEl.lastElementChild;
            div.querySelector(".mapper-count").value = n;
            div.querySelector(".mapper-check").checked = checked;
        });
    }


    function gatherSelected() {
        return Array.from(listEl.querySelectorAll(".mapper-item"))
            // NEW: keep only the rows that are checked
            .filter(div => div.querySelector(".mapper-check").checked)
            .map(div => ({
                id: div.dataset.id,
                name: div.querySelector(".mapper-name").value.trim(),
                n: parseInt(div.querySelector(".mapper-count").value, 10) || 1,
            }));
    }

    listEl.addEventListener("click", (e) => {
        if (e.target.classList.contains("remove-mapper-btn")) {
            e.target.closest(".mapper-item").remove();
        }
    });

    return { addMapper, gatherSelected, getAll, clearAll, loadFromArray };

})();

/* ------------------------------------------------------------------ */
/* Glue into existing buttons */

document.getElementById("add-mapper-btn").onclick = () => {
    const id = document.getElementById("new_mapper_id").value;
    document.getElementById("new_mapper_id").value = "";
    MapperManager.addMapper(id);
};

document.getElementById("add-to-queue-btn").onclick = () => {
    /* ── NEW: ensure placeholders are resolved & a song is loaded ── */
    PathManager.applyPlaceholderValues?.();           // harmless noop if not defined

    const audio = document.getElementById("audio_path").value.trim();
    const beat = document.getElementById("beatmap_path").value.trim();
    if (!audio && !beat) {
        Utils?.showFlashMessage("Load an audio or beatmap first.", "error");
        return;                                       // abort adding a blank task
    }
    /* ────────────────────────────────────────────────────────────── */

    const mappers = MapperManager.gatherSelected();
    const fd = new FormData(document.getElementById("inferenceForm"));

    /* fall‑back entry when nothing is ticked */
    (mappers.length ? mappers : [{ id: "", name: "", n: 1 }]).forEach(mp => {
        for (let i = 0; i < mp.n; i++) {

            const artist = fd.get("artist")?.trim() || "??";
            const title = fd.get("title")?.trim() || "??";

            /* ------------ decide mapper/creator ---------------- */
            const uiName = fd.get("mapper_name")?.trim();          // manual override field
            const creator = uiName || mp.name || `Mapperatorinator ${fd.get("model")?.toUpperCase()}`;

            /* ------------ difficulty string -------------------- */
            let diffName = fd.get("difficulty_name")?.trim();
            if (!diffName) {
                const stars = parseFloat(fd.get("difficulty")) || 3.5;
                const baseName = _defaultDiffLabel(stars);
                diffName = (mp.id || mp.name) ? `${creator}'s ${baseName}` : baseName;
            }

            /* ------------ assemble task ------------------------ */
            const task = Object.fromEntries(fd.entries());
            task.mapper_id = mp.id;
            task.mapper_display_name = mp.name;
            task.artist = artist;
            task.title = title;
            task.creator = creator;
            task.difficulty_string = diffName;
            task.display_name = `${artist} - ${title} (${creator}) [${diffName}]`;

            QueueManager.add(task);
            Utils?.showFlashMessage(`Queued ${task.display_name}`, 'success');
            // QueueManager.start();
        }
    });
};



/* ------------------------------------------------------------------ */
/* Thin wrapper util expected by QueueManager */

const InferenceManager = {
    async runTask(task) {
        // POST form data → /start_inference (same as original code)
        const formData = new FormData();
        Object.entries(task).forEach(([k, v]) => formData.append(k, v));
        // Here you can reuse your existing JS that triggers SSE etc.
        await window.startInferenceWithFormData(formData); // you must implement
    }
};

/* initial render */
QueueManager.render();

window.queueAPI = {
    hasJobs: () => QueueManager.hasPending(),
    isRunning: () => QueueManager.isRunning(),
    start: () => QueueManager.start(),
    clear: () => QueueManager.clear()
};
