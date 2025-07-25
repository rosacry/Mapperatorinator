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

            // Add status indicator
            const status = (idx === 0 && running) ?
                '<span class="status-running">● Running</span> ' :
                '<span class="status-pending">○ Pending</span> ';

            li.innerHTML = status + `${task.display_name}...`;

            const del = document.createElement("button");
            del.textContent = "✕";
            del.className = "delete-btn";
            del.onclick = () => remove(idx);
            li.appendChild(del);
            list.appendChild(li);
        });
    }

    /* ---------- public helpers ---------- */
    function add(task) { queue.push(task); render(); }
    function clear() { if (!running) { queue.length = 0; render(); } }
    function stop() { running = false; queue.length = 0; render(); }
    async function remove(i) {
        if (running && i === 0) {
            // Cancel current inference but keep the queue running
            await InferenceManager.cancelInference();

            // Remove only this item
            queue.splice(i, 1);
            render();

            // Immediately start next item if exists
            if (queue.length > 0) {
                _runNext();
            } else {
                running = false;
            }
        } else {
            // For non-running items, just remove them
            queue.splice(i, 1);
            render();
        }
    }
    function hasPending() { return queue.length > 0; }
    function isRunning() { return running; }

    /* ---------- execution loop ---------- */
    async function _runNext() {
        if (!queue.length) { running = false; render(); return; }
        running = true;
        const task = queue[0];              // grab first
        render();
        try {
            window._queueInProgress = true;  // ➌ tell handleSubmit who called
            await InferenceManager.runTask(task);
        } catch (e) {
            console.error("Task failed", e);
        } finally {
            queue.shift();                   // ➍ drop finished job
            window._queueInProgress = false;
            _runNext();                      //   …and start the next
        }
    }
    function start() { if (!running && queue.length) _runNext(); }

    return { add, clear, remove, stop, hasPending, isRunning, start, render };
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
            // Add this to explicitly include mapper_id in form data
            // if (mp.id) {
            //     formData.append('mapper_id', mp.id);
            // }
            task.mapper_display_name = mp.name;
            task.mapper_name = mp.name;
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

// Update the runTask function
// Replace the existing runTask function with this:
const InferenceManager = {
    async runTask(task) {
        // Create proper form data with mapper_id
        const formData = new FormData();
        Object.entries(task).forEach(([k, v]) => {
            // Special handling for mapper_id
            if (k === "mapper_id" && v) {
                formData.append("mapper_id", v);
            }
            formData.append(k, v);
        });

        // POST to inference endpoint
        await window.startInferenceWithFormData(formData);
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
