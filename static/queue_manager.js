/* queue_manager.js – handles generation queue + mapper list */

function _defaultDiffLabel(stars) {
    if (stars < 2.5) return "Easy";
    else if (stars < 3.5) return "Normal";
    else if (stars < 4.5) return "Hard";
    else if (stars < 5.5) return "Insane";
    else return "Expert";
}

const QueueManager = (() => {
    const queue = [];
    let running = false;

    function render() {
        const list = document.getElementById("queue-list");
        list.innerHTML = "";
        queue.forEach((task, idx) => {
            const li = document.createElement("li");
            li.classList.add("queue-item");
            li.textContent = `${task.audio_path} → ${task.output_path} (mapper ${task.mapper_display_name || "—"})`;
            const del = document.createElement("button");
            del.textContent = "✕";
            del.className = "delete-btn";
            del.onclick = () => { remove(idx); };
            li.appendChild(del);
            if (idx === 0 && running) li.classList.add("running");
            list.appendChild(li);
        });
        // const btn = document.getElementById("add-to-queue-btn");
        // if (btn) btn.disabled = running;
    }

    function add(task) {
        queue.push(task);
        render();
    }
    function remove(i) {
        if (running && i === 0) return;      // can't remove current
        queue.splice(i, 1);
        render();
    }
    function hasPending() { return queue.length > 0; }

    async function _runNext() {
        if (!queue.length) { running = false; render(); return; }
        running = true; render();
        const task = queue[0];
        try {
            await InferenceManager.runTask(task);   // defined below
        } catch (e) {
            console.error("Task failed", e);
        } finally {
            queue.shift();            // drop finished
            _runNext();               // recurse
        }
    }
    function start() {                 // kick off processing from outside
        if (!running && queue.length) _runNext();
    }

    return {
        add, remove, hasPending, start,
        /* Queue now starts automatically when the first item is added. */
        markFinished() { /* called by InferenceManager */ },
        render
    };
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
    const mappers = MapperManager.gatherSelected();
    const fd = new FormData(document.getElementById("inferenceForm"));

    /* fall-back entry when nothing is ticked */
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

                /* include "<mapper>'s" only when sampling */
                if (mp.id || mp.name) {
                    diffName = `${creator}'s ${baseName}`;
                } else {
                    diffName = baseName;
                }
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
    start: () => QueueManager.start()
};