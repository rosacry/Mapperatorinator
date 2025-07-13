/* queue_manager.js – handles generation queue + mapper list */

const QueueManager = (() => {
    const queue = [];
    let running = false;

    function render() {
        const list = document.getElementById("queue-list");
        list.innerHTML = "";
        queue.forEach((task, idx) => {
            const li = document.createElement("li");
            li.textContent = `${task.audio_path} → ${task.output_path} (mapper ${task.mapper_display_name || "—"})`;
            const del = document.createElement("button");
            del.textContent = "✕";
            del.className = "delete-btn";
            del.onclick = () => { remove(idx); };
            li.appendChild(del);
            if (idx === 0 && running) li.classList.add("running");
            list.appendChild(li);
        });
        document.getElementById("start-queue-btn").disabled = running || queue.length === 0;
        document.getElementById("add-to-queue-btn").disabled = running;
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

    return {
        add, remove, hasPending,
        start() { if (!running && queue.length) _runNext(); },
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

    function gatherSelected() {
        return Array.from(listEl.querySelectorAll(".mapper-item"))
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

    return { addMapper, gatherSelected };
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
    if (mappers.length === 0) {
        alert("Select at least one mapper (checkbox) first.");
        return;
    }
    // collect form -> plain object
    const fd = new FormData(document.getElementById("inference-form"));
    mappers.forEach(mp => {
        for (let i = 0; i < mp.n; i++) {
            const t = Object.fromEntries(fd.entries());
            t.mapper_id = mp.id;
            t.mapper_display_name = mp.name;
            QueueManager.add(t);
        }
    });
};

document.getElementById("start-queue-btn").onclick = () => {
    QueueManager.start();
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
