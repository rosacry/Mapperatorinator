$(document).ready(function () {
    // Application state and configuration
    const AppState = {
        jobs: new Map(),
        jobCounter: 0,
        lastStartedJobId: null,
        animationSpeed: 300,

        modelCapabilities: {
            "v28": {},
            "v29": {},
            "v30": {
                supportedGamemodes: ['0'],
                supportsYear: false,
                supportedInContextOptions: ['TIMING'],
                hideHitsoundsOption: true,
                supportsDescriptors: false,
            },
        }
    };

    // Utility functions
    const Utils = {
        showFlashMessage(message, type = 'success') {
            const flashContainer = $('#flash-container');
            const alertClass = type === 'success' ? 'alert success' :
                type === 'cancel-success' ? 'alert alert-cancel-success' :
                    'alert error';
            const messageDiv = $(`<div class="${alertClass}">${message}</div>`);
            flashContainer.append(messageDiv);
            setTimeout(() => messageDiv.remove(), 5000);
        },

        smoothScroll(target, offset = 0) {
            $('html, body').animate({
                scrollTop: $(target).offset().top + offset
            }, 500);
        },

        resetFormToDefaults() {
            $('#inferenceForm')[0].reset();

            // Clear descriptors
            $('input[name="descriptors"], input[name="in_context_options"]')
                .removeClass('positive-check negative-check').prop('checked', false);

            ValidationManager.clearPlaceholders();
            return ValidationManager.validateAndAutofill(false);
        }
    };

    // UI Manager for conditional visibility
    const UIManager = {
        clearable_inputs: '#audio_path, #beatmap_path, #output_path, #lora_path, #background_image',

        init() {
            this.attachClearButtonHandlers();
            $(this.clearable_inputs).trigger('blur');
        },

        attachClearButtonHandlers() {
            // Listen for input events (typing)
            $(this.clearable_inputs).on('input', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Listen for blur events (leaving field) - immediate validation
            $(this.clearable_inputs).on('blur', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Handle clear button clicks
            $('.clear-input-btn').on('click', (e) => {
                const targetId = $(e.target).data('target');
                const $targetInput = $(`#${targetId}`);

                $targetInput.val('');
                this.updateClearButtonVisibility($targetInput[0]);
                return ValidationManager.validateAndAutofill(false);
            });
        },

        updateClearButtonVisibility(inputElement) {
            const $input = $(inputElement);
            const $clearBtn = $input.siblings('.clear-input-btn');
            const hasValue = $input.val().trim() !== '';

            if (hasValue) {
                $clearBtn.show();
            } else {
                $clearBtn.hide();
            }
        },

        updateConditionalFields() {
            const selectedGamemode = $("#gamemode").val();
            const selectedModel = $("#model").val();
            const beatmapPath = $('#beatmap_path').val().trim();

            // Handle gamemode-based visibility
            $('.conditional-field[data-show-for-gamemode]').each(function () {
                const $field = $(this);
                const supportedModes = $field.data('show-for-gamemode').toString().split(',');
                const shouldShow = supportedModes.includes(selectedGamemode);

                if (shouldShow && !$field.is(':visible')) {
                    $field.slideDown(AppState.animationSpeed);
                } else if (!shouldShow && $field.is(':visible')) {
                    $field.slideUp(AppState.animationSpeed);
                }
            });

            // Handle model-based visibility
            $('.conditional-field[data-hide-for-model]').each(function () {
                const $field = $(this);
                const hiddenModels = $field.data('hide-for-model').toString().split(',');
                const shouldHide = hiddenModels.includes(selectedModel);

                if (shouldHide && $field.is(':visible')) {
                    $field.slideUp(AppState.animationSpeed);
                } else if (!shouldHide && !$field.is(':visible')) {
                    $field.slideDown(AppState.animationSpeed);
                }
            });

            // Handle beatmap path dependent fields
            const shouldShowBeatmapFields = beatmapPath !== '';
            ['#in-context-options-box', '#add-to-beatmap-option', '#overwrite-reference-beatmap-option'].forEach(selector => {
                const $element = $(selector);
                if (shouldShowBeatmapFields && !$element.is(':visible')) {
                    $element.fadeIn(AppState.animationSpeed);
                } else if (!shouldShowBeatmapFields && $element.is(':visible')) {
                    $element.fadeOut(AppState.animationSpeed);
                    if (selector === '#add-to-beatmap-option') {
                        $('#add_to_beatmap').prop('checked', false);
                    }
                    if (selector === '#overwrite-reference-beatmap-option') {
                        $('#overwrite_reference_beatmap').prop('checked', false);
                    }
                }
            });
        },

        updateModelSettings() {
            const selectedModel = $("#model").val();
            const capabilities = AppState.modelCapabilities[selectedModel] || {};

            // Handle gamemode restrictions
            const $gamemodeSelect = $("#gamemode");
            if (selectedModel === "v30") {
                $gamemodeSelect.val('0').prop('disabled', true);
                $gamemodeSelect.find("option").each(function () {
                    $(this).prop('disabled', $(this).val() !== '0');
                });
            } else {
                $gamemodeSelect.prop('disabled', false);
                $gamemodeSelect.find("option").prop('disabled', false);
            }

            // Handle in-context options
            const supportedContext = capabilities.supportedInContextOptions ||
                ['NONE', 'TIMING', 'KIAI', 'MAP', 'GD', 'NO_HS'];

            $('input[name="in_context_options"]').each(function () {
                const $checkbox = $(this);
                const value = $checkbox.val();
                const $item = $checkbox.closest('.context-option-item');
                const isSupported = supportedContext.includes(value);

                $item.data('model-allowed', isSupported);
                $checkbox.prop('disabled', !isSupported);

                if (isSupported) {
                    $item.slideDown(AppState.animationSpeed);
                } else {
                    $item.slideUp(AppState.animationSpeed);
                }
            });

            // Handle hitsounds for V30
            if (capabilities.hideHitsoundsOption) {
                $('#hitsounded').prop('checked', true);
            }

            this.updateConditionalFields();
        }
    };

    // File Browser Manager
    const FileBrowser = {
        init() {
            this.attachBrowseHandlers();
        },

        attachBrowseHandlers() {
            $('.browse-button[data-browse-type]').click(async function () {
                const browseType = $(this).data('browse-type');
                const targetId = $(this).data('target');

                try {
                    let path;

                    if (browseType === 'folder') {
                        path = await window.pywebview.api.browse_folder();
                    } else if (browseType === 'image') {
                        path = await window.pywebview.api.browse_image();
                    } else {
                        let fileTypes = null;

                        if (targetId === 'beatmap_path') {
                            fileTypes = [
                                'Beatmap Files (*.osu)',
                                'All files (*.*)'
                            ];
                        } else if (targetId === 'audio_path') {
                            fileTypes = [
                                // todo: add more formats if needed and implement this in backend as well + add error msgs
                                'Audio Files (*.mp3;*.wav;*.ogg;*.m4a;*.flac)',
                                'All files (*.*)'
                            ];
                        }

                        path = await window.pywebview.api.browse_file(fileTypes);
                    }

                    if (path) {
                        if (targetId === 'beatmap_path' && !path.toLowerCase().endsWith('.osu')) {
                            Utils.showFlashMessage('Please select a valid .osu file.', 'error');
                            // Set the path and let validation handle inline error
                        }

                        const $targetInput = $(`#${targetId}`);
                        $targetInput.val(path);
                        console.log(`Selected ${browseType}:`, path);

                        // Trigger input event to update clear buttons and validate
                        $targetInput.trigger('input');
                        $targetInput.trigger('blur'); // Trigger blur to validate
                    }
                } catch (error) {
                    console.error(`Error browsing for ${browseType}:`, error);
                    alert(`Could not browse for ${browseType}. Ensure the backend API is running.`);
                }
            });
        }
    };

    const validation_trigger_inputs = '#audio_path, #beatmap_path, #output_path';

    // Path Manager for autofill, validation and clear button support
    const ValidationManager = {
        init() {
            this.attachValidationChangeHandlers();
            $(validation_trigger_inputs).trigger('blur');
        },

        attachValidationChangeHandlers() {
            // Listen for blur events (leaving field) - immediate validation
            $(validation_trigger_inputs).on('blur', (_) => {
                return this.validateAndAutofill(false);
            });
        },

        validateAndAutofill(showFlashMessages = false) { // isFileDialog replaced by showFlashMessages
            const audioPath = $('#audio_path').val().trim();
            const beatmapPath = $('#beatmap_path').val().trim();
            const outputPath = $('#output_path').val().trim();

            // Call backend validation
            return new Promise((resolve) => {
                $.ajax({
                    url: '/validate_paths',
                    method: 'POST',
                    data: {
                        audio_path: audioPath,
                        beatmap_path: beatmapPath,
                        output_path: outputPath
                    },
                    success: (response) => {
                        this.handleValidationResponse(response, showFlashMessages);
                        resolve(response.success);
                    },
                    error: (xhr, status, error) => {
                        console.error('Path validation failed:', error);
                        if (showFlashMessages) {
                            Utils.showFlashMessage('Error validating paths. Check console for details.', 'error');
                        }
                        this.clearPlaceholders();
                        resolve(false);
                    }
                });
            });
        },

        placeholder_elements: {
            '#audio_path': 'audio_path',
            '#output_path': 'output_path',
            '#beatmap_path': 'beatmap_path',
            '#gamemode': 'gamemode',
            '#difficulty': 'difficulty',
            '#title': 'title',
            '#title_unicode': 'title_unicode',
            '#artist': 'artist',
            '#artist_unicode': 'artist_unicode',
            '#creator': 'creator',
            '#version': 'version',
            '#preview_time': 'preview_time',
            '#background_image': 'background',
            '#source': 'source',
            '#tags': 'tags',
            '#hp_drain_rate': 'hp_drain_rate',
            '#circle_size': 'circle_size',
            '#approach_rate': 'approach_rate',
            '#overall_difficulty': 'overall_difficulty',
            '#slider_multiplier': 'slider_multiplier',
            '#slider_tick_rate': 'slider_tick_rate',
            '#hold_note_ratio': 'hold_note_ratio',
            '#scroll_speed_ratio': 'scroll_speed_ratio',
            '#mapper_id': 'mapper_id',
        },

        handleValidationResponse(response, showFlashMessages = false) {
            this.clearValidationErrors();
            const autofilledArgs = response.autofilled_args;

            // Show autofilled values as placeholders
            Object.entries(this.placeholder_elements).forEach(([selector, argName]) => {
                const $input = $(selector);
                if (autofilledArgs && autofilledArgs[argName] !== undefined && autofilledArgs[argName] !== null) {
                    $input.attr('placeholder', autofilledArgs[argName]);
                } else {
                    $input.attr('placeholder', '');
                }
            });

            if (showFlashMessages) {
                // Show errors as flash messages and inline indicators
                response.errors.forEach(error => {
                    Utils.showFlashMessage(error, 'error');
                });
            }

            // Always show/update inline errors
            response.errors.forEach(error => {
                this.showInlineErrorForMessage(error);
            });

            // Update UI for conditional fields
            UIManager.updateConditionalFields();
        },

        showInlineErrorForMessage(error) {
            const audioPathVal = $('#audio_path').val().trim();
            const beatmapPathVal = $('#beatmap_path').val().trim();

            if (error.includes('Audio file not found') && (audioPathVal || beatmapPathVal)) {
                this.showInlineError('#audio_path', 'Audio file not found');
            } else if (error.includes('Beatmap file not found') && beatmapPathVal) {
                this.showInlineError('#beatmap_path', 'Beatmap file not found');
            } else if (error.includes('Beatmap file must have .osu extension') && beatmapPathVal) {
                this.showInlineError('#beatmap_path', 'Must be .osu file');
            }
        },

        showInlineError(inputSelector, message) {
            const $input = $(inputSelector);
            const $inputContainer = $input.closest('.input-with-clear');
            // Prevent duplicate error messages
            if ($input.siblings('.path-validation-error').length > 0) {
                $input.siblings('.path-validation-error').text(message);
            } else {
                const $errorDiv = $(`<div class="path-validation-error" style="color: #ff4444; font-size: 12px; margin-top: 2px;">${message}</div>`);
                $inputContainer.after($errorDiv);
            }
        },

        clearValidationErrors() {
            $('.path-validation-error').remove();
        },

        clearPlaceholders() {
            Object.keys(this.placeholder_elements).forEach(selector => {
                $(selector).attr('placeholder', '');
            });
            this.clearValidationErrors();
        },
    };

    // Descriptor Manager
    const DescriptorManager = {
        init() {
            this.attachDropdownHandler();
            this.attachDescriptorClickHandlers();
        },

        attachDropdownHandler() {
            $('.custom-dropdown-descriptors .dropdown-header').on('click', function () {
                const $dropdown = $(this).parent();
                const dropdownContent = document.querySelector('.dropdown-content');
                $dropdown.toggleClass('open');
                if ($dropdown.hasClass('open')) {
                    Utils.smoothScroll('.custom-dropdown-descriptors');
                    dropdownContent.removeAttribute('inert');
                } else {
                    dropdownContent.setAttribute('inert', '');
                }
            });
        },

        attachDescriptorClickHandlers() {
            $('.descriptors-container').on('click', 'input[name="descriptors"]', function (e) {
                e.preventDefault();
                const $checkbox = $(this);

                if (!$checkbox.prop('disabled')) {
                    if ($checkbox.hasClass('positive-check')) {
                        $checkbox.removeClass('positive-check').addClass('negative-check');
                    } else if ($checkbox.hasClass('negative-check')) {
                        $checkbox.removeClass('negative-check');
                        $checkbox.prop('checked', false);
                        return;
                    } else {
                        $checkbox.addClass('positive-check');
                    }
                    $checkbox.prop('checked', true);
                }
            });
        }
    };

    // Configuration Manager
    const ConfigManager = {
        init() {
            $('#export-config-btn').click(() => this.exportConfiguration());
            $('#import-config-btn').click(() => $('#import-config-input').click());
            $('#reset-config-btn').click(() => this.resetToDefaults());
            $('#import-config-input').change((e) => this.handleFileImport(e));
        },

        exportConfiguration() {
            const config = this.buildConfigObject();

            if (window.pywebview?.api?.save_file) {
                this.exportToFile(config);
            } else {
                this.fallbackDownload(config);
            }
        },

        buildConfigObject() {
            const config = {
                version: "1.1",
                timestamp: new Date().toISOString(),
                settings: {},
                descriptors: { positive: [], negative: [] },
                inContextOptions: [],
                mapperList: [],
                songMetadata: { artist: '', title: '' },
                beatmapCustomization: { previewTime: '', backgroundPath: '' }
            };

            // Export form fields
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    config.settings[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Export descriptors
            $('input[name="descriptors"]').each(function () {
                const $checkbox = $(this);
                const value = $checkbox.val();
                if ($checkbox.hasClass('positive-check')) {
                    config.descriptors.positive.push(value);
                } else if ($checkbox.hasClass('negative-check')) {
                    config.descriptors.negative.push(value);
                }
            });

            // Export in-context options
            $('input[name="in_context_options"]:checked').each(function () {
                config.inContextOptions.push($(this).val());
            });

            // Export mapper list
            if (typeof MapperManager !== 'undefined') {
                config.mapperList = MapperManager.getAll();
            }

            // Export song metadata
            config.songMetadata.artist = $('#artist').val() || '';
            config.songMetadata.title = $('#title').val() || '';

            // Export beatmap customization
            config.beatmapCustomization.previewTime = $('#preview_time').val() || '';
            config.beatmapCustomization.backgroundPath = $('#background_image').val() || '';

            return config;
        },

        async exportToFile(config) {
            try {
                const filename = `mapperatorinator-config-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;

                const filePath = await window.pywebview.api.save_file(filename);
                if (!filePath) {
                    this.showConfigStatus("Export cancelled by user", "error");
                    return;
                }

                $.ajax({
                    url: "/save_config",
                    method: "POST",
                    data: {
                        file_path: filePath,
                        config_data: JSON.stringify(config, null, 2)
                    },
                    success: (response) => {
                        if (response.success) {
                            this.showConfigStatus(`Configuration exported successfully to: ${response.file_path}`, "success");
                        } else {
                            this.showConfigStatus(`Error saving config: ${response.error}`, "error");
                        }
                    },
                    error: () => {
                        this.showConfigStatus("Failed to save config to server. Using browser download instead.", "error");
                        this.fallbackDownload(config);
                    }
                });
            } catch (error) {
                console.error("Error selecting folder:", error);
                this.fallbackDownload(config);
            }
        },

        fallbackDownload(config) {
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mapperatorinator-config-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            this.showConfigStatus("Configuration exported successfully (browser download)", "success");
        },

        resetToDefaults() {
            if (confirm("Are you sure you want to reset all settings to default values? This cannot be undone.")) {
                Utils.resetFormToDefaults();
                $("#model, #gamemode, #beatmap_path").trigger('change');
                $(UIManager.clearable_inputs).trigger('blur');
                this.showConfigStatus("All settings reset to default values", "success");
            }
        },

        handleFileImport(e) {
            const file = e.target.files[0];
            if (!file) return;

            if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
                this.showConfigStatus("Please select a valid JSON configuration file.", "error");
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => this.importConfiguration(e.target.result);
            reader.readAsText(file);
            $(e.target).val(''); // Reset input
        },

        // Map legacy field names from older config versions to current form field names
        _fieldNameMap: {
            'background_path': 'background_image',
            'diff_name': 'version',
            'detected_artist': null,  // skip, handled via songMetadata
            'detected_title': null,   // skip, handled via songMetadata
        },

        importConfiguration(content) {
            try {
                const config = JSON.parse(content);
                if (!config.version) {
                    throw new Error("Invalid configuration file format");
                }

                // Import settings with field name mapping
                if (config.settings) {
                    Object.entries(config.settings).forEach(([name, value]) => {
                        // Apply field name mapping
                        const mapped = this._fieldNameMap.hasOwnProperty(name) ? this._fieldNameMap[name] : name;
                        if (mapped === null) return; // Explicitly skipped field
                        const $field = $(`[name="${mapped}"]`);
                        if ($field.length) {
                            if ($field.attr('type') === 'checkbox') {
                                $field.prop('checked', !!value);
                            } else {
                                $field.val(value);
                            }
                        }
                    });
                }

                // Import descriptors
                $('input[name="descriptors"]').removeClass('positive-check negative-check').prop('checked', false);
                if (config.descriptors) {
                    config.descriptors.positive?.forEach(value => {
                        $(`input[name="descriptors"][value="${value}"]`)
                            .addClass('positive-check').prop('checked', true);
                    });
                    config.descriptors.negative?.forEach(value => {
                        $(`input[name="descriptors"][value="${value}"]`)
                            .addClass('negative-check').prop('checked', true);
                    });
                }

                // Import in-context options
                $('input[name="in_context_options"]').prop('checked', false);
                config.inContextOptions?.forEach(value => {
                    $(`input[name="in_context_options"][value="${value}"]`).prop('checked', true);
                });

                // Import mapper list
                if (config.mapperList && Array.isArray(config.mapperList) && config.mapperList.length > 0) {
                    if (typeof MapperManager !== 'undefined') {
                        MapperManager.loadFromArray(config.mapperList);
                    }
                }

                // Import song metadata
                if (config.songMetadata) {
                    if (config.songMetadata.artist) $('#artist').val(config.songMetadata.artist);
                    if (config.songMetadata.title) $('#title').val(config.songMetadata.title);
                }

                // Import beatmap customization
                if (config.beatmapCustomization) {
                    if (config.beatmapCustomization.previewTime) {
                        $('#preview_time').val(config.beatmapCustomization.previewTime);
                        if (typeof BeatmapCustomization !== 'undefined') BeatmapCustomization.updatePreviewDisplay();
                    }
                    if (config.beatmapCustomization.backgroundPath) {
                        $('#background_image').val(config.beatmapCustomization.backgroundPath);
                    }
                }

                // Trigger updates
                $("#model, #gamemode").trigger('change');
                $(UIManager.clearable_inputs).trigger('blur');
                $(UIManager.clearable_inputs).trigger('input');
                if (typeof QueueUI !== 'undefined') QueueUI.updateUI();

                this.showConfigStatus(`Configuration imported successfully! (${config.timestamp || 'Unknown date'})`, "success");

            } catch (error) {
                console.error("Error importing configuration:", error);
                this.showConfigStatus(`Error importing configuration: ${error.message}`, "error");
            }
        },

        showConfigStatus(message, type) {
            const $status = $("#config-status");
            $status.text(message)
                .css('color', type === 'success' ? '#28a745' : '#dc3545')
                .fadeIn();
            setTimeout(() => $status.fadeOut(), 5000);
        }
    };

    // Inference Manager
    const InferenceManager = {
        init() {
            $('#inferenceForm').submit((e) => this.handleSubmit(e));
        },

        async handleSubmit(e) {
            e.preventDefault();

            // Apply placeholder values before validation
            if (!await this.validateForm()) return;

            this.removeFinishedCards();
            const formData = this.buildFormData();

            // Determine job label suffix based on title/title_unicode/audio filename
            const jobLabelSuffix = this.getJobLabelSuffix(formData);
            const job = this.createJobCard(jobLabelSuffix);
            this.startInference(job, formData);
        },

        async validateForm() {
            const $audioPath = $('#audio_path');
            const $beatmapPath = $('#beatmap_path');
            const $outputPath = $('#output_path');

            const audioPath = $audioPath.val().trim() || $audioPath.attr('placeholder');
            const beatmapPath = $beatmapPath.val().trim() || $beatmapPath.attr('placeholder');
            const outputPath = $outputPath.val().trim() || $outputPath.attr('placeholder');

            if (!audioPath && !beatmapPath) {
                Utils.smoothScroll(0);
                Utils.showFlashMessage("Either 'Beatmap Path' or 'Audio Path' are required for running inference", 'error');
                return false;
            }

            if (!outputPath && !beatmapPath) {
                Utils.smoothScroll(0);
                Utils.showFlashMessage("Either 'Output Path' or 'Beatmap Path' are required for running inference", 'error');
                return false;
            }

            // Validate beatmap file type if beatmap path is provided
            if (beatmapPath && !beatmapPath.toLowerCase().endsWith('.osu')) {
                Utils.smoothScroll('#beatmap_path');
                Utils.showFlashMessage("Beatmap file must have .osu extension", 'error');
                ValidationManager.showInlineError('#beatmap_path', 'Must be .osu file');
                return false;
            }

            const pathsAreValid = await ValidationManager.validateAndAutofill(true);
            if (!pathsAreValid) {
                Utils.smoothScroll(0);
                return false;
            }

            return true;
        },

        createJobCard(labelSuffix = "") {
            AppState.jobCounter += 1;
            // Build job display name with suffix when available
            const baseName = `Job ${AppState.jobCounter}`;
            const jobDisplayName = labelSuffix ? `${baseName} - ${labelSuffix}` : baseName;
            const tempKey = `temp-${Date.now()}-${AppState.jobCounter}`;

            const $card = $(
                `<div class="progress-card" data-status="running" data-job-key="${tempKey}">
                    <div class="progress-card-header">
                        <div class="progress-card-title">${jobDisplayName}</div>
                        <button type="button" class="progress-card-close" title="Remove">×</button>
                    </div>
                    <div class="progress-card-status">Starting...</div>
                    <div class="warning-text" style="display:none; font-size: 12px; color: var(--accent-color); margin-top: 4px;">
                        Warning on this job (Will continue to generate)
                    </div>
                    <div class="init-message" style="font-style: italic; color: #ccc; margin-bottom: 10px;">
                        Initializing process... This may take a moment.
                    </div>
                    <div class="progressBarContainer">
                        <div class="progressBar"></div>
                    </div>
                    <div class="progress-card-actions">
                        <button type="button" class="cancel-button" style="display:none;">Cancel</button>
                    </div>
                    <div class="progress-card-links warning-log-link" style="display:none;">
                        <a href="#">View warning log</a>
                    </div>
                    <pre class="warning-log" style="display:none; white-space: pre-wrap; background: #141414; border: 1px solid var(--border-color); padding: 8px; border-radius: 6px; margin-top: 8px;"></pre>
                    <div class="progress-card-links beatmap-link" style="display:none;">
                        <a href="#" target="_blank">Click here to open the folder containing your map.</a>
                    </div>
                    <div class="progress-card-links error-log-link" style="display:none;">
                        <a href="#">See why... (opens error log)</a>
                    </div>
                </div>`
            );

            $('#progressCards').prepend($card);
            $('#progress_output').show();
            Utils.smoothScroll('#progress_output');

            const job = {
                id: null,
                tempKey,
                displayName: jobDisplayName,
                stage: 'starting',
                errorIndicatorSeen: false,
                warningMessages: [],
                warningCaptureActive: false,
                warningCaptureRemaining: 0,
                warningSuppressed: false,
                evtSource: null,
                isCancelled: false,
                inferenceErrorOccurred: false,
                accumulatedErrorMessages: [],
                errorLogFilePath: null,
                elements: {
                    $card,
                    $status: $card.find('.progress-card-status'),
                    $warningText: $card.find('.warning-text'),
                    $initMessage: $card.find('.init-message'),
                    $progressBar: $card.find('.progressBar'),
                    $progressBarContainer: $card.find('.progressBarContainer'),
                    $cancelButton: $card.find('.cancel-button'),
                    $warningLogLink: $card.find('.warning-log-link'),
                    $warningLogLinkAnchor: $card.find('.warning-log-link a'),
                    $warningLog: $card.find('.warning-log'),
                    $beatmapLink: $card.find('.beatmap-link'),
                    $beatmapLinkAnchor: $card.find('.beatmap-link a'),
                    $errorLogLink: $card.find('.error-log-link'),
                    $errorLogLinkAnchor: $card.find('.error-log-link a')
                }
            };

            $card.find('.progress-card-close').on('click', () => this.requestClose(job, $card));
            job.elements.$cancelButton.on('click', () => this.requestCancel(job));

            AppState.jobs.set(tempKey, job);
            return job;
        },

        removeJob(jobId, $cardOverride = null) {
            const job = this.getJob(jobId);
            const $card = $cardOverride || job?.elements?.$card;
            if (job?.evtSource) {
                job.evtSource.close();
            }
            if ($card) {
                const tempKey = $card.data('job-key');
                $card.remove();
                if (!jobId && tempKey) {
                    AppState.jobs.delete(tempKey);
                }
            }
            if (job) {
                AppState.jobs.delete(job.id || job.tempKey);
            }
            if (AppState.lastStartedJobId && job && AppState.lastStartedJobId === job.id) {
                AppState.lastStartedJobId = null;
            }
            this.updateProgressOutputVisibility();
        },

        removeFinishedCards() {
            $('.progress-card').each((_, card) => {
                const $card = $(card);
                const status = $card.data('status');
                if (status === 'completed' || status === 'error' || status === 'cancelled') {
                    const jobId = $card.data('job-id');
                    const tempKey = $card.data('job-key');
                    $card.remove();
                    if (jobId) {
                        AppState.jobs.delete(jobId);
                    } else if (tempKey) {
                        AppState.jobs.delete(tempKey);
                    }
                }
            });
            this.updateProgressOutputVisibility();
        },

        updateProgressOutputVisibility() {
            if ($('#progressCards').children().length === 0) {
                $('#progress_output').hide();
            }
        },

        buildFormData() {
            const formData = new FormData($("#inferenceForm")[0]);

            // Handle descriptors
            formData.delete('descriptors');
            const positiveDescriptors = [];
            const negativeDescriptors = [];

            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) {
                    positiveDescriptors.push($cb.val());
                } else if ($cb.hasClass('negative-check')) {
                    negativeDescriptors.push($cb.val());
                }
            });

            positiveDescriptors.forEach(val => formData.append('descriptors', val));
            negativeDescriptors.forEach(val => formData.append('negative_descriptors', val));

            // Ensure hitsounded is true for V30
            if ($("#model").val() === "v30" && !$("#option-item-hitsounded").is(':visible')) {
                formData.set('hitsounded', 'true');
            }

            return formData;
        },

        // Compute job label suffix from Title (Unicode), Title, or audio filename
        getJobLabelSuffix(formData) {
            const maxLength = 60;

            const sanitizeLabel = (value) => {
                const text = (value || '').toString().replace(/\s+/g, ' ').trim();
                if (!text) return '';
                return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
            };

            const titleUnicode = sanitizeLabel(formData.get('title_unicode'));
            if (titleUnicode) return titleUnicode;

            const title = sanitizeLabel(formData.get('title'));
            if (title) return title;

            const audioPathRaw = (formData.get('audio_path') || '').toString().trim();
            if (audioPathRaw) {
                // Normalize path separators and strip any trailing separators
                const normalized = audioPathRaw.replace(/\\/g, '/').replace(/\/+$/, '');
                const filename = normalized.split('/').pop();
                const safeFilename = sanitizeLabel(filename);
                if (safeFilename) return safeFilename;
            }

            return '';
        },

        startInference(job, formData) {
            $.ajax({
                url: "/start_inference",
                method: "POST",
                data: formData,
                processData: false,
                contentType: false,
                success: (response) => {
                    const jobId = response.job_id;
                    if (!jobId) {
                        Utils.showFlashMessage("Failed to start inference: missing job id.", 'error');
                        this.removeJob(job.id || job.tempKey, job.elements.$card);
                        return;
                    }
                    job.id = jobId;
                    job.elements.$cancelButton.show().prop('disabled', false).text('Cancel');
                    job.elements.$card.attr('data-job-id', jobId);
                    AppState.jobs.delete(job.tempKey);
                    AppState.jobs.set(jobId, job);
                    AppState.lastStartedJobId = jobId;
                    this.connectToSSE(job);
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    console.error("Failed to start inference:", textStatus, errorThrown);
                    let errorMsg = "Failed to start inference process. Check backend console.";
                    if (jqXHR.responseJSON && jqXHR.responseJSON.message) {
                        errorMsg = jqXHR.responseJSON.message;
                    } else if (jqXHR.responseText) {
                        try {
                            const parsed = JSON.parse(jqXHR.responseText);
                            if (parsed && parsed.message) errorMsg = parsed.message;
                        } catch (e) { /* ignore parsing error */ }
                    }
                    Utils.showFlashMessage(errorMsg, 'error');
                    this.removeJob(job.id, job.elements.$card);
                }
            });
        },

        connectToSSE(job) {
            console.log("Connecting to SSE stream...", job.id);
            job.evtSource = new EventSource(`/stream_output?job_id=${encodeURIComponent(job.id)}`);
            job.errorLogFilePath = null;

            job.evtSource.onmessage = (e) => this.handleSSEMessage(job, e);
            job.evtSource.onerror = (err) => this.handleSSEError(job, err);
            job.evtSource.addEventListener("error_log", (e) => {
                job.errorLogFilePath = e.data;
            });
            job.evtSource.addEventListener("end", (e) => this.handleSSEEnd(job, e));
        },

        handleSSEMessage(job, e) {
            if (job.elements.$initMessage.is(":visible")) job.elements.$initMessage.hide();
            if (job.isCancelled) return;

            const messageData = e.data;
            const errorIndicators = [
                "Traceback (most recent call last):", "Error executing job with overrides:",
                "FileNotFoundError:", "Exception:", "Set the environment variable HYDRA_FULL_ERROR=1"
            ];

            const isErrorMessage = errorIndicators.some(indicator => messageData.includes(indicator));
            const isClientDisconnectTrace = messageData.includes("_client_handler") ||
                messageData.includes("Exception in thread Thread-") ||
                messageData.includes("GeneratorExit") ||
                messageData.includes("connection_dropped") ||
                messageData.includes("generator ignored GeneratorExit") ||
                messageData.includes("BrokenPipeError") ||
                messageData.includes("The pipe is being closed") ||
                messageData.includes("[WinError 232]");

            if (job.warningCaptureActive) {
                job.warningMessages.push(messageData);
                job.warningCaptureRemaining -= 1;
                if (job.warningCaptureRemaining <= 0) {
                    job.warningCaptureActive = false;
                }
                if (isClientDisconnectTrace) {
                    job.warningSuppressed = true;
                }
                if (job.warningSuppressed) {
                    job.warningMessages = [];
                    job.warningCaptureActive = false;
                    job.elements.$warningLog.hide().text('');
                    job.elements.$warningLogLink.hide();
                    job.elements.$warningText.hide();
                    return;
                }
                job.elements.$warningLog.text(job.warningMessages.join("\n"));
            }

            if (isErrorMessage && !isClientDisconnectTrace) {
                job.errorIndicatorSeen = true;
                job.accumulatedErrorMessages.push(messageData);
                job.warningMessages.push(messageData);
                job.warningCaptureActive = true;
                job.warningCaptureRemaining = 80;
                job.elements.$warningText.show();
                job.elements.$warningLog.text(job.warningMessages.join("\n"));
                if (job.elements.$warningLogLink.is(':hidden')) {
                    job.elements.$warningLogLinkAnchor.off("click").on("click", (event) => {
                        event.preventDefault();
                        job.elements.$warningLog.toggle();
                    });
                    job.elements.$warningLogLink.show();
                }
            } else if (job.inferenceErrorOccurred) {
                job.accumulatedErrorMessages.push(messageData);
            } else {
                this.updateProgress(job, messageData);
            }
        },

        updateProgress(job, messageData) {
            // Update progress title based on message content
            const lowerCaseMessage = messageData.toLowerCase();
            const progressTitles = {
                "generating timing": "Generating Timing",
                "generating kiai": "Generating Kiai",
                "generating map": "Generating Map",
                "seq len": "Refining Positions"
            };

            Object.entries(progressTitles).forEach(([keyword, title]) => {
                if (lowerCaseMessage.includes(keyword)) {
                    job.elements.$status.text(title);
                    if (job.stage !== 'generating') {
                        job.stage = 'generating';
                    }
                }
            });

            // Update progress bar
            const progressMatch = messageData.match(/^\s*(\d+)%\|/);
            if (progressMatch) {
                const currentPercent = parseInt(progressMatch[1].trim(), 10);
                if (!isNaN(currentPercent)) {
                    job.elements.$progressBar.css("width", currentPercent + "%");
                }
            }

            // Check for completion message
            if (messageData.includes("Generated beatmap saved to")) {
                const parts = messageData.split("Generated beatmap saved to");
                if (parts.length > 1) {
                    const fullPath = parts[1].trim().replace(/\\/g, "/");
                    const folderPath = fullPath.substring(0, fullPath.lastIndexOf("/"));

                    job.elements.$beatmapLinkAnchor
                        .attr("href", "#")
                        .text("Click here to open the folder containing your map.")
                        .off("click")
                        .on("click", (e) => {
                            e.preventDefault();
                            $.get("/open_folder", { folder: folderPath })
                                .done(response => console.log("Open folder response:", response))
                                .fail(() => alert("Failed to open folder via backend."));
                        });
                    job.elements.$beatmapLink.show();
                }
            }
        },

        handleSSEError(job, err) {
            console.error("EventSource failed:", err);
            if (job.evtSource) {
                job.evtSource.close();
                job.evtSource = null;
            }

            job.stage = 'finished';

            if (!job.isCancelled && !job.inferenceErrorOccurred) {
                job.inferenceErrorOccurred = true;
                job.accumulatedErrorMessages.push("Error: Connection to process stream lost.");
                job.elements.$status.text("Connection Error").css('color', 'var(--accent-color)');
                job.elements.$progressBar.addClass('error');
                job.elements.$card.data('status', 'error');
                Utils.showFlashMessage("Error: Connection to process stream lost.", "error");
            }

            job.elements.$cancelButton.hide();
        },

        handleSSEEnd(job, e) {
            console.log("Received end event from server.", e.data);
            if (job.evtSource) {
                job.evtSource.close();
                job.evtSource = null;
            }

            const endMessage = (e.data || '').toLowerCase();
            const endWithErrors = endMessage.includes('with errors');
            if (endWithErrors) {
                job.inferenceErrorOccurred = true;
            }

            if (job.isCancelled) {
                job.elements.$status.text("Cancelled").css('color', 'var(--accent-color)');
                job.elements.$progressBar.addClass('error');
                job.elements.$card.data('status', 'cancelled');
            } else if (job.inferenceErrorOccurred) {
                job.warningMessages = [];
                job.warningCaptureActive = false;
                job.warningSuppressed = false;
                job.elements.$warningLog.hide().text('');
                job.elements.$warningLogLink.hide();
                job.elements.$warningText.hide();
                this.handleInferenceError(job);
                job.elements.$card.data('status', 'error');
            } else {
                job.elements.$status.text("Processing Complete").css('color', '');
                job.elements.$progressBar.css("width", "100%").removeClass('error');
                job.elements.$card.data('status', 'completed');
            }

            job.elements.$cancelButton.hide();
            job.isCancelled = false;
        },

        handleInferenceError(job) {
            const fullErrorText = job.accumulatedErrorMessages.join("\\n");
            let specificError = "An error occurred during processing. Check console/logs.";

            if (fullErrorText.includes("FileNotFoundError:")) {
                const fileNotFoundMatch = fullErrorText.match(/FileNotFoundError:.*? file (.*?) not found/);
                specificError = fileNotFoundMatch?.[1] ?
                    `Error: File not found - ${fileNotFoundMatch[1].replace(/\\\\/g, '\\\\')}` :
                    "Error: A required file was not found.";
            } else if (fullErrorText.includes("HYDRA_FULL_ERROR=1")) {
                specificError = "There was an error while creating the beatmap. Check console/logs for details.";
            } else if (fullErrorText.includes("Error executing job")) {
                specificError = "There was an error starting or executing the generation task.";
            } else if (fullErrorText.includes("Connection to process stream lost")) {
                specificError = "Error: Connection to the generation process was lost.";
            }

            Utils.showFlashMessage(specificError, "error");
            job.elements.$status.text("Processing Failed").css('color', 'var(--accent-color)').show();
            job.elements.$progressBar.css("width", "100%").addClass('error');
            job.elements.$beatmapLink.hide();

            if (job.errorLogFilePath) {
                job.elements.$errorLogLinkAnchor.off("click").on("click", (e) => {
                    e.preventDefault();
                    $.get("/open_log_file", { path: job.errorLogFilePath })
                        .done(response => console.log("Open log response:", response))
                        .fail(() => alert("Failed to open log file via backend."));
                });
                job.elements.$errorLogLink.show();
            }
        },

        requestCancel(job) {
            this.cancelInference(job);
        },

        cancelInference(job) {
            const $cancelBtn = job.elements.$cancelButton;
            $cancelBtn.prop('disabled', true).text('Cancelling...');

            $.ajax({
                url: "/cancel_inference",
                method: "POST",
                data: { job_id: job.id },
                success: (response) => { // Expecting JSON response
                    job.isCancelled = true;
                    Utils.showFlashMessage(response.message || "Inference cancelled successfully.", "cancel-success");
                },
                error: (jqXHR) => {
                    const errorMsg = jqXHR.responseJSON?.message || "Failed to send cancel request. Unknown error.";
                    Utils.showFlashMessage(errorMsg, "error");
                    $cancelBtn.prop('disabled', false).text('Cancel');
                }
            });
        },

        requestClose(job, $card) {
            const status = $card?.data('status');
            if (job.stage === 'finished' || status === 'completed' || status === 'error' || status === 'cancelled') {
                this.removeJob(job.id || job.tempKey, $card);
                return;
            }
            this.cancelInference(job);
            this.removeJob(job.id || job.tempKey, $card);
        },

        getJob(jobId) {
            return AppState.jobs.get(jobId) || null;
        }
    };

    // ========================================
    // USER FEATURE MODULES (Queue, Mapper, Preview, Detection)
    // ========================================

    // Beatmap Customization Manager (preview time picker, background preview)
    const BeatmapCustomization = {
        audioElement: null,
        lastAudioPath: '',
        lastPickerPosition: 0,

        init() {
            this.attachEventHandlers();
            this.setupBackgroundPreview();
            this.updatePreviewDisplay();
        },

        attachEventHandlers() {
            $('#pick-preview-btn').on('click', () => this.openPreviewPicker());
            $('#preview_time').on('input', () => this.updatePreviewDisplay());
            $('#background_path, #background_image').on('input blur', () => this.updateBackgroundPreview());
            $('#audio_path').on('change blur', () => this.onAudioPathChanged());
        },

        setupBackgroundPreview() {
            $(document).on('click', '.clear-input-btn[data-target="background_path"], .clear-input-btn[data-target="background_image"]', () => {
                setTimeout(() => this.updateBackgroundPreview(), 10);
            });
        },

        onAudioPathChanged() {
            const currentAudioPath = $('#audio_path').val().trim();
            if (this.lastAudioPath && currentAudioPath !== this.lastAudioPath) {
                this.clearCustomizations();
            }
            this.lastAudioPath = currentAudioPath;
        },

        clearCustomizations() {
            $('#preview_time').val('');
            $('#preview-time-display').text('').removeClass('has-value');
            this.lastPickerPosition = 0;
            $('#background_path, #background_image').val('');
            $('#background-preview').hide();
        },

        updatePreviewDisplay() {
            const ms = parseInt($('#preview_time').val());
            const $display = $('#preview-time-display');
            if (!isNaN(ms) && ms >= 0) {
                const minutes = Math.floor(ms / 60000);
                const secs = Math.floor((ms % 60000) / 1000);
                $display.text(`(${minutes}:${secs.toString().padStart(2, '0')})`).addClass('has-value');
            } else {
                $display.text('').removeClass('has-value');
            }
        },

        updateBackgroundPreview() {
            const bgPath = ($('#background_path').val() || $('#background_image').val() || '').trim();
            const $preview = $('#background-preview');
            const $img = $('#background-preview-img');
            if (!bgPath) { $preview.hide(); return; }
            $.ajax({
                url: '/get_image_preview', method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ path: bgPath }),
                success: (response) => {
                    if (response.success && response.data) {
                        $img.attr('src', 'data:image/' + response.type + ';base64,' + response.data);
                        $preview.show();
                    } else { $preview.hide(); }
                },
                error: () => { $preview.hide(); }
            });
        },

        openPreviewPicker() {
            const audioPath = $('#audio_path').val().trim() || $('#audio_path').attr('placeholder');
            if (!audioPath) { Utils.showFlashMessage('Please select an audio file first.', 'error'); return; }
            this.createPreviewModal(audioPath);
        },

        createPreviewModal(audioPath) {
            $('#preview-picker-modal').remove();
            const modalHtml = `
                <div id="preview-picker-modal" class="preview-modal-overlay">
                    <div class="preview-modal">
                        <div class="preview-modal-header">
                            <h3>Pick Preview Point</h3>
                            <button type="button" class="preview-modal-close">×</button>
                        </div>
                        <div class="preview-modal-body">
                            <div class="preview-controls-row">
                                <label>Playback Speed:</label>
                                <select id="preview-speed">
                                    <option value="0.25">25%</option>
                                    <option value="0.5">50%</option>
                                    <option value="0.75">75%</option>
                                    <option value="1" selected>100%</option>
                                </select>
                            </div>
                            <div class="preview-slider-container" id="preview-slider-track">
                                <div class="preview-slider-fill" id="preview-slider-fill"></div>
                                <div class="preview-slider-thumb" id="preview-slider-thumb"></div>
                            </div>
                            <div class="preview-time-labels">
                                <span id="preview-current-time">0:00</span>
                                <span id="preview-total-time">--:--</span>
                            </div>
                            <div class="preview-input-row">
                                <div class="preview-input-item">
                                    <label>Milliseconds:</label>
                                    <input type="number" id="preview-ms-input" min="0" value="0" />
                                </div>
                                <div class="preview-input-item">
                                    <label>Seconds:</label>
                                    <input type="number" id="preview-sec-input" min="0" step="1" value="0" />
                                </div>
                            </div>
                            <div class="preview-volume-row">
                                <label>Volume:</label>
                                <div class="volume-slider-container" id="volume-slider-track">
                                    <div class="volume-slider-fill" id="volume-slider-fill"></div>
                                    <div class="volume-slider-thumb" id="volume-slider-thumb"></div>
                                </div>
                            </div>
                            <div class="preview-buttons-row">
                                <button type="button" id="preview-play-btn" class="browse-button">Play / Pause</button>
                                <button type="button" id="preview-test-btn" class="browse-button" title="Play 10 seconds from current position">Test Preview</button>
                                <button type="button" id="preview-set-btn" class="browse-button accent">Use This Point</button>
                            </div>
                        </div>
                    </div>
                </div>`;
            $('body').append(modalHtml);
            this.setupPreviewAudio(audioPath);
        },

        setupPreviewAudio(audioPath) {
            $.ajax({
                url: '/get_audio_info', method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ path: audioPath }),
                success: (response) => {
                    if (response.success) {
                        this.initializeAudioPlayer(response.url || audioPath, response.duration);
                    } else {
                        Utils.showFlashMessage('Failed to load audio: ' + (response.message || 'Unknown error'), 'error');
                        $('#preview-picker-modal').remove();
                    }
                },
                error: () => {
                    this.initializeAudioPlayer('file:///' + audioPath.replace(/\\/g, '/'));
                }
            });
            $('.preview-modal-close, .preview-modal-overlay').on('click', (e) => {
                if (e.target === e.currentTarget) this.closePreviewModal();
            });
            $('.preview-modal').on('click', (e) => e.stopPropagation());
        },

        initializeAudioPlayer(audioUrl, duration) {
            if (this.audioElement) { this.audioElement.pause(); this.audioElement = null; }
            this.audioElement = new Audio();
            this.audioElement.crossOrigin = 'anonymous';
            const self = this;
            const $sliderTrack = $('#preview-slider-track'), $sliderFill = $('#preview-slider-fill'), $sliderThumb = $('#preview-slider-thumb');
            const $volumeTrack = $('#volume-slider-track'), $volumeFill = $('#volume-slider-fill'), $volumeThumb = $('#volume-slider-thumb');
            const $msInput = $('#preview-ms-input'), $secInput = $('#preview-sec-input');
            const $currentTime = $('#preview-current-time'), $totalTime = $('#preview-total-time'), $speed = $('#preview-speed');
            const existingPreviewTime = parseInt($('#preview_time').val()) || 0;
            const initialPosition = existingPreviewTime > 0 ? existingPreviewTime : self.lastPickerPosition;
            this.audioElement.volume = 0.5;
            $volumeFill.css('width', '50%'); $volumeThumb.css('left', '50%');
            self.sliderDragging = false; self.audioLoaded = false; self.audioDuration = 0; self.currentMs = 0;

            function updateSliderVisual(ms) {
                if (self.audioDuration <= 0) return;
                const percent = Math.min(100, Math.max(0, (ms / self.audioDuration) * 100));
                $sliderFill.css('width', percent + '%'); $sliderThumb.css('left', percent + '%');
            }
            function updateDisplays(ms) {
                self.currentMs = ms; $msInput.val(Math.floor(ms)); $secInput.val(Math.floor(ms / 1000));
                $currentTime.text(self.formatTime(ms)); updateSliderVisual(ms);
            }
            function seekAudioTo(ms) {
                const targetSec = Math.max(0, Math.min(ms / 1000, self.audioElement.duration || 0));
                self.audioElement.currentTime = targetSec;
            }

            this.audioElement.addEventListener('loadedmetadata', () => {
                self.audioDuration = self.audioElement.duration * 1000;
                $msInput.attr('max', Math.floor(self.audioDuration));
                $secInput.attr('max', Math.floor(self.audioDuration / 1000));
                $totalTime.text(self.formatTime(self.audioDuration));
                if (initialPosition > 0 && initialPosition <= self.audioDuration) {
                    updateDisplays(initialPosition); seekAudioTo(initialPosition);
                } else { updateDisplays(0); }
                self.audioLoaded = true;
            });
            this.audioElement.addEventListener('timeupdate', () => {
                if (!self.sliderDragging && self.audioLoaded) updateDisplays(self.audioElement.currentTime * 1000);
            });

            function handleSliderInteraction(e) {
                if (!self.audioLoaded || self.audioDuration <= 0) return 0;
                const rect = $sliderTrack[0].getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const newMs = Math.floor(percent * self.audioDuration);
                updateDisplays(newMs); return newMs;
            }

            let wasPlaying = false;
            $sliderTrack.on('mousedown', function (e) {
                e.preventDefault(); e.stopPropagation(); self.sliderDragging = true;
                wasPlaying = !self.audioElement.paused;
                if (wasPlaying) self.audioElement.pause();
                handleSliderInteraction(e);
                function onMouseMove(moveEvent) { handleSliderInteraction(moveEvent); }
                function onMouseUp() {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    seekAudioTo(self.currentMs);
                    if (wasPlaying) {
                        const onSeeked = () => { self.audioElement.removeEventListener('seeked', onSeeked); self.audioElement.play(); self.sliderDragging = false; };
                        self.audioElement.addEventListener('seeked', onSeeked);
                        setTimeout(() => { self.audioElement.removeEventListener('seeked', onSeeked); if (self.audioElement.paused && wasPlaying) self.audioElement.play(); self.sliderDragging = false; }, 300);
                    } else { self.sliderDragging = false; }
                }
                document.addEventListener('mousemove', onMouseMove); document.addEventListener('mouseup', onMouseUp);
            });

            function handleVolumeInteraction(e) {
                const rect = $volumeTrack[0].getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                self.audioElement.volume = percent;
                $volumeFill.css('width', (percent * 100) + '%'); $volumeThumb.css('left', (percent * 100) + '%');
            }
            $volumeTrack.on('mousedown', function (e) {
                e.preventDefault(); handleVolumeInteraction(e);
                function onMouseMove(moveEvent) { handleVolumeInteraction(moveEvent); }
                function onMouseUp() { document.removeEventListener('mousemove', onMouseMove); document.removeEventListener('mouseup', onMouseUp); }
                document.addEventListener('mousemove', onMouseMove); document.addEventListener('mouseup', onMouseUp);
            });

            $msInput.on('change', () => { const ms = parseInt($msInput.val()) || 0; if (self.audioLoaded) { updateDisplays(ms); seekAudioTo(ms); } });
            $secInput.on('change', () => { const ms = (parseInt($secInput.val()) || 0) * 1000; if (self.audioLoaded) { updateDisplays(ms); seekAudioTo(ms); } });
            $speed.on('change', () => { self.audioElement.playbackRate = parseFloat($speed.val()); });
            $('#preview-play-btn').on('click', () => {
                if (self.audioElement.paused) { if (self.audioLoaded) { seekAudioTo(self.currentMs); self.audioElement.play(); } else { self.audioElement.play(); } }
                else { self.audioElement.pause(); }
            });

            let testPreviewTimeout = null, savedPosition = 0, isTestingPreview = false;
            $('#preview-test-btn').on('click', () => {
                if (isTestingPreview) {
                    clearTimeout(testPreviewTimeout); self.audioElement.pause();
                    seekAudioTo(savedPosition); updateDisplays(savedPosition);
                    isTestingPreview = false; $('#preview-test-btn').text('Test Preview'); return;
                }
                savedPosition = self.currentMs; isTestingPreview = true; $('#preview-test-btn').text('Stop Test');
                seekAudioTo(savedPosition); self.audioElement.play();
                testPreviewTimeout = setTimeout(() => {
                    self.audioElement.pause(); seekAudioTo(savedPosition); updateDisplays(savedPosition);
                    isTestingPreview = false; $('#preview-test-btn').text('Test Preview');
                }, 10000);
            });

            $('#preview-set-btn').on('click', () => {
                const ms = parseInt($msInput.val()) || 0;
                $('#preview_time').val(ms); self.updatePreviewDisplay(); self.closePreviewModal();
                Utils.showFlashMessage(`Preview point set to ${self.formatTime(ms)}`, 'success');
            });

            this.audioElement.src = audioUrl; this.audioElement.load();
        },

        formatTime(ms) {
            const totalSeconds = Math.floor(ms / 1000);
            return `${Math.floor(totalSeconds / 60)}:${(totalSeconds % 60).toString().padStart(2, '0')}`;
        },

        closePreviewModal() {
            const currentMs = parseInt($('#preview-ms-input').val()) || 0;
            if (currentMs > 0) this.lastPickerPosition = currentMs;
            if (this.audioElement) { this.audioElement.pause(); this.audioElement = null; }
            $('#preview-picker-modal').remove();
        }
    };

    // Mapper Name Lookup
    const MapperLookup = {
        cache: {},
        pendingLookups: {},

        init() {
            $('#mapper_id').on('blur', () => this.lookupCurrentMapper());
            $('#mapper_id').on('input', () => this.clearDisplay());
        },

        clearDisplay() {
            $('#mapper_name_display').removeClass('visible loading error').text('');
            $('#mapper_name').val('');
        },

        async lookupCurrentMapper() {
            const mapperId = $('#mapper_id').val().trim();
            if (!mapperId) { this.clearDisplay(); return; }
            const $display = $('#mapper_name_display');
            if (this.cache[mapperId]) {
                $display.text(`(${this.cache[mapperId]})`).removeClass('loading error').addClass('visible');
                $('#mapper_name').val(this.cache[mapperId]); return;
            }
            $display.text('Looking up...').removeClass('error visible').addClass('loading visible');
            try {
                const name = await this.lookup(mapperId);
                if (name) {
                    this.cache[mapperId] = name;
                    $display.text(`(${name})`).removeClass('loading error').addClass('visible');
                    $('#mapper_name').val(name);
                } else {
                    $display.text('(Not found)').removeClass('loading visible').addClass('error visible');
                    $('#mapper_name').val('');
                }
            } catch (error) {
                $display.text('(Lookup failed)').removeClass('loading visible').addClass('error visible');
                $('#mapper_name').val('');
            }
        },

        async lookup(mapperId) {
            if (this.pendingLookups[mapperId]) return this.pendingLookups[mapperId];
            this.pendingLookups[mapperId] = new Promise((resolve, reject) => {
                $.ajax({
                    url: '/lookup_mapper_name', method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ mapper_id: mapperId }),
                    success: (response) => { delete this.pendingLookups[mapperId]; resolve(response.username || null); },
                    error: (xhr) => { delete this.pendingLookups[mapperId]; reject(new Error(xhr.responseJSON?.error || 'Lookup failed')); }
                });
            });
            return this.pendingLookups[mapperId];
        }
    };

    // Difficulty Name Generator
    const DifficultyNameGenerator = {
        init() {
            $('#difficulty').on('input change', () => this.updateDifficultyName());
            $('#auto_generate_diff_name').on('change', () => {
                if ($('#auto_generate_diff_name').prop('checked')) this.updateDifficultyName();
            });
        },
        getDifficultyName(stars) {
            stars = parseFloat(stars) || 5.0;
            if (stars < 2.0) return 'Easy'; if (stars < 2.7) return 'Normal';
            if (stars < 4.0) return 'Hard'; if (stars < 5.3) return 'Insane';
            if (stars < 6.5) return 'Expert'; return 'Expert+';
        },
        updateDifficultyName() {
            if (!$('#auto_generate_diff_name').prop('checked')) return;
            $('#version').val(this.getDifficultyName($('#difficulty').val()));
        }
    };

    // Song Detection Manager (audio fingerprinting)
    const SongDetection = {
        lastAudioPath: '', isDetecting: false, detectionCancelled: false,

        init() {
            $('#audio_path').on('blur', () => this.detectSongInfo());
        },

        async detectSongInfo() {
            if (!$('#auto_detect_metadata').prop('checked')) return;
            const audioPath = $('#audio_path').val().trim();
            if (!audioPath || audioPath === this.lastAudioPath) return;
            this.lastAudioPath = audioPath;

            const $status = $('#song-detection-status');
            this.isDetecting = true;
            $status.html('Detecting song info... <span class="detection-cancel" title="Cancel detection">×</span>')
                .removeClass('success error').addClass('detecting').show();
            $status.find('.detection-cancel').on('click', (e) => { e.stopPropagation(); this.cancelDetection(); });
            $('#add-to-queue-btn').prop('disabled', true).addClass('detecting');
            $('button[type="submit"]').prop('disabled', true);
            if (typeof QueueUI !== 'undefined') QueueUI.updateButtons();
            this.detectionCancelled = false;

            try {
                const response = await $.ajax({
                    url: '/validate_paths', method: 'POST',
                    data: { audio_path: audioPath, beatmap_path: '', output_path: '', detect_song: true }
                });
                if (this.detectionCancelled) return;
                if (response.detected_artist) $('#artist').val(response.detected_artist);
                if (response.detected_title) $('#title').val(response.detected_title);
                if (response.detected_artist || response.detected_title) {
                    $status.text('✓ Song detected!').addClass('success');
                } else {
                    $status.text('Could not detect song info automatically.').addClass('error');
                }
            } catch (error) {
                if (this.detectionCancelled) return;
                $status.text('Detection failed.').addClass('error');
            } finally {
                if (!this.detectionCancelled) {
                    this.isDetecting = false; $status.removeClass('detecting');
                    $('#add-to-queue-btn').removeClass('detecting');
                    $('button[type="submit"]').prop('disabled', false);
                    if (typeof QueueUI !== 'undefined') QueueUI.updateButtons();
                }
            }
        },

        forceDetect() { this.lastAudioPath = ''; this.detectSongInfo(); },

        cancelDetection() {
            if (!this.isDetecting) return;
            this.isDetecting = false; this.detectionCancelled = true;
            $('#auto_detect_metadata').prop('checked', false);
            $('#song-detection-status').text('Detection cancelled').removeClass('detecting').addClass('error');
            $('#add-to-queue-btn').removeClass('detecting').prop('disabled', false);
            $('button[type="submit"]').prop('disabled', false);
            if (typeof QueueUI !== 'undefined') QueueUI.updateButtons();
        },

        isInProgress() { return this.isDetecting; }
    };

    // Queue UI Manager — integrates with QueueManager (from queue_manager.js) and upstream InferenceManager
    const QueueUI = {
        queueRunning: false,
        currentRunningTaskId: null,
        totalTasksAtStart: 0,
        completedTasksForProgress: 0,

        init() {
            if (typeof QueueManager === 'undefined') {
                console.warn('QueueManager not loaded. Queue features disabled.');
                $('#queue_panel, #mapper_panel').hide();
                return;
            }
            this.attachEventHandlers();
            this.updateUI();
        },

        attachEventHandlers() {
            $('#add-to-queue-btn').on('click', () => this.addCurrentFormToQueue());
            $('#run-queue-btn').on('click', () => this.runQueue());
            $('#clear-queue-btn').on('click', () => this.clearQueue());
            $('#add-mapper-btn').off('click').on('click', () => this.addMapper());
            $('#generate-from-mappers-btn').off('click').on('click', () => this.generateTasksFromMappers());
            window.addEventListener('queueStateChanged', () => this.updateUI());
            $('#audio_path').on('input blur change', () => this.updateButtons());
            $('#add-mapper-id').off('keypress').on('keypress', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); this.addMapper(); }
            });
        },

        addCurrentFormToQueue() {
            if (SongDetection.isInProgress()) {
                Utils.showFlashMessage('Please wait for song detection to complete.', 'error'); return;
            }
            const formData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this), name = $field.attr('name'), type = $field.attr('type');
                if (name && type !== 'file') {
                    formData[name] = type === 'checkbox' ? $field.prop('checked') : ($field.val() || $field.attr('placeholder') || '');
                }
            });
            formData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) formData.descriptors.positive.push($cb.val());
                else if ($cb.hasClass('negative-check')) formData.descriptors.negative.push($cb.val());
            });
            const mapperName = $('#mapper_name').val() || '', mapperId = formData.mapper_id || '';
            formData.mapper_name = mapperName;
            const artist = formData.artist || '??', title = formData.title || '??';
            const model = formData.model || 'v30', creator = mapperName || `Mapperatorinator ${model.toUpperCase()}`;
            const stars = parseFloat(formData.difficulty) || 5.0;
            const diffName = formData.version || DifficultyNameGenerator.getDifficultyName(stars);
            const baseDisplayName = `${artist} - ${title} (${creator}) [${diffName}]`;
            const existingTasks = QueueManager.getAllTasks();
            let count = 0;
            existingTasks.forEach(task => {
                const n = task.formData.display_name || '';
                if (n === baseDisplayName || n.match(new RegExp(`^${baseDisplayName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(\\d+\\)$`))) count++;
            });
            formData.display_name = count > 0 ? `${baseDisplayName} (${count})` : baseDisplayName;
            QueueManager.addTask(formData);
            this.updateUI();
            Utils.showFlashMessage('Task added to queue!', 'success');
        },

        async runQueue() {
            if (QueueManager.isEmpty()) { Utils.showFlashMessage('Queue is empty!', 'error'); return; }
            await $.ajax({ url: '/reset_queue', method: 'POST' });
            const compileAsBeatmapSet = $('#compile-as-beatmapset').prop('checked');
            if (compileAsBeatmapSet) {
                await $.ajax({ url: '/init_beatmapset', method: 'POST', contentType: 'application/json', data: JSON.stringify({ enabled: true }) });
            }
            $('#run-queue-btn, #clear-queue-btn').prop('disabled', true);
            $('#compile-as-beatmapset').prop('disabled', true);
            this.queueRunning = true;
            this.totalTasksAtStart = QueueManager.getAllTasks().length;
            this.completedTasksForProgress = 0;
            $('#queue_progress_info').show();
            this.updateOverallQueueProgress(0, 0);

            const tasks = QueueManager.getAllTasks();
            let completedCount = 0, queueCancelled = false;

            for (let i = 0; i < tasks.length; i++) {
                const statusResponse = await $.ajax({ url: '/queue_status', method: 'GET' });
                if (statusResponse.cancelled) { queueCancelled = true; break; }
                const task = tasks[i];
                QueueManager.setTaskStatus(task.id, 'running');
                this.currentRunningTaskId = task.id;
                this.updateUI();
                try {
                    await this.runSingleTask(task);
                    const postStatus = await $.ajax({ url: '/queue_status', method: 'GET' });
                    if (postStatus.cancelled) { queueCancelled = true; QueueManager.setTaskStatus(task.id, 'pending'); break; }
                    QueueManager.setTaskStatus(task.id, 'completed');
                    completedCount++;
                    this.completedTasksForProgress++;
                    this.updateOverallQueueProgress(this.completedTasksForProgress, 100);
                } catch (error) {
                    const errorStatus = await $.ajax({ url: '/queue_status', method: 'GET' });
                    if (errorStatus.cancelled) { queueCancelled = true; QueueManager.setTaskStatus(task.id, 'pending'); break; }
                    QueueManager.setTaskStatus(task.id, error.message === 'Task skipped' ? 'skipped' : 'failed');
                    this.completedTasksForProgress++;
                    this.updateOverallQueueProgress(this.completedTasksForProgress, 100);
                }
                this.updateUI();
            }

            this.queueRunning = false; this.currentRunningTaskId = null;
            $('#run-queue-btn, #clear-queue-btn').prop('disabled', false);
            $('#compile-as-beatmapset').prop('disabled', false);
            setTimeout(() => { if (!this.queueRunning) $('#queue_progress_info').hide(); }, 3000);
            this.updateUI();

            if (!queueCancelled) {
                if (compileAsBeatmapSet && completedCount > 0) {
                    try {
                        const response = await $.ajax({ url: '/finalize_beatmapset', method: 'POST' });
                        Utils.showFlashMessage(response.success ?
                            `Queue completed: ${completedCount}/${tasks.length} tasks. Beatmap set: ${response.filename}` :
                            `Queue completed: ${completedCount}/${tasks.length} tasks. Set creation failed.`, response.success ? 'success' : 'error');
                    } catch (e) { Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks.`, 'success'); }
                } else {
                    Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks.`, 'success');
                }
            }
        },

        runSingleTask(task) {
            return new Promise((resolve, reject) => {
                const formData = new FormData();
                Object.entries(task.formData).forEach(([key, value]) => {
                    if (key === 'descriptors') {
                        value.positive?.forEach(v => formData.append('descriptors', v));
                        value.negative?.forEach(v => formData.append('negative_descriptors', v));
                    } else if (typeof value === 'boolean') {
                        formData.append(key, value.toString());
                    } else if (value !== null && value !== undefined && value !== '') {
                        formData.append(key, value);
                    }
                });

                // Use upstream InferenceManager's job card system
                const jobLabel = task.formData.display_name || `Queue Task`;
                const job = InferenceManager.createJobCard(jobLabel);

                // Override the SSE end handler to resolve/reject the promise
                const originalHandleSSEEnd = InferenceManager.handleSSEEnd.bind(InferenceManager);
                const taskJobId = { value: null };

                // Start inference via upstream's method
                $.ajax({
                    url: "/start_inference", method: "POST", data: formData,
                    processData: false, contentType: false,
                    success: (response) => {
                        const jobId = response.job_id;
                        if (!jobId) { InferenceManager.removeJob(job.tempKey); reject(new Error('No job_id')); return; }
                        job.id = jobId;
                        taskJobId.value = jobId;
                        job.elements.$cancelButton.show().prop('disabled', false).text('Cancel');
                        job.elements.$card.attr('data-job-id', jobId);
                        AppState.jobs.delete(job.tempKey);
                        AppState.jobs.set(jobId, job);
                        AppState.lastStartedJobId = jobId;

                        // Connect to SSE with completion tracking
                        job.evtSource = new EventSource(`/stream_output?job_id=${encodeURIComponent(jobId)}`);
                        job.evtSource.onmessage = (e) => {
                            InferenceManager.handleSSEMessage(job, e);
                            const percentMatch = e.data.match(/^\s*(\d+)%\|/);
                            if (percentMatch) {
                                this.updateOverallQueueProgress(this.completedTasksForProgress, parseInt(percentMatch[1]));
                            }
                        };
                        job.evtSource.onerror = (err) => {
                            InferenceManager.handleSSEError(job, err);
                            reject(new Error('Connection lost'));
                        };
                        job.evtSource.addEventListener("error_log", (e) => { job.errorLogFilePath = e.data; });
                        job.evtSource.addEventListener("end", (e) => {
                            originalHandleSSEEnd(job, e);
                            if (job.isCancelled || job.inferenceErrorOccurred) {
                                reject(new Error(job.isCancelled ? 'Task skipped' : 'Inference failed'));
                            } else {
                                resolve();
                            }
                        });
                    },
                    error: (jqXHR) => {
                        InferenceManager.removeJob(job.tempKey);
                        reject(new Error(jqXHR.responseJSON?.message || 'Failed to start task'));
                    }
                });
            });
        },

        updateOverallQueueProgress(completedTasks, currentTaskProgress) {
            const totalTasks = this.totalTasksAtStart || 1;
            const overallPercent = ((completedTasks * 100) + currentTaskProgress) / (totalTasks * 100) * 100;
            $('#queue_task_label').text(`Queue: ${completedTasks}/${totalTasks} tasks (${Math.round(overallPercent)}%)`);
        },

        cancelQueue() {
            $.ajax({
                url: '/cancel_inference', method: 'POST',
                contentType: 'application/json', data: JSON.stringify({ clear_queue: true }),
                success: (response) => {
                    $('#progress_output, #queue_progress_info').hide();
                    if (this.currentRunningTaskId) { QueueManager.setTaskStatus(this.currentRunningTaskId, 'pending'); this.updateUI(); }
                    $('#compile-as-beatmapset').prop('disabled', false);
                    Utils.showFlashMessage(response.message || 'Queue cancelled.', 'cancel-success');
                },
                error: (xhr) => { Utils.showFlashMessage(xhr.responseJSON?.message || 'Failed to cancel queue.', 'error'); }
            });
        },

        skipCurrentTask() {
            $.ajax({
                url: '/cancel_inference', method: 'POST',
                contentType: 'application/json', data: JSON.stringify({ clear_queue: false }),
            });
        },

        clearQueue() {
            if (QueueManager.isEmpty()) return;
            QueueManager.clearAll();
            this.updateUI();
            Utils.showFlashMessage('Queue cleared.', 'success');
        },

        _addingMapper: false,
        async addMapper() {
            if (this._addingMapper) return;
            const mapperId = $('#add-mapper-id').val().trim();
            if (!mapperId) return;
            this._addingMapper = true;
            $('#add-mapper-id').val('');
            try {
                let mapperName = MapperLookup.cache[mapperId];
                if (!mapperName) { try { mapperName = await MapperLookup.lookup(mapperId); } catch (e) { Utils.showFlashMessage('User not found.', 'error'); return; } }
                if (!mapperName) { Utils.showFlashMessage('User not found.', 'error'); return; }
                MapperManager.addMapper(mapperId, mapperName, 1);
                this.updateUI();
                Utils.showFlashMessage(`Mapper added: ${mapperName}`, 'success');
            } finally { this._addingMapper = false; }
        },

        addSingleMapperToQueue(mapperId, mapperName, count) {
            const templateData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this), name = $field.attr('name'), type = $field.attr('type');
                if (name && type !== 'file') templateData[name] = type === 'checkbox' ? $field.prop('checked') : ($field.val() || $field.attr('placeholder') || '');
            });
            templateData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) templateData.descriptors.positive.push($cb.val());
                else if ($cb.hasClass('negative-check')) templateData.descriptors.negative.push($cb.val());
            });
            const artist = templateData.artist || '??', title = templateData.title || '??';
            const model = templateData.model || 'v30', stars = parseFloat(templateData.difficulty) || 5.0;
            const baseDiff = DifficultyNameGenerator.getDifficultyName(stars);
            let tasksAdded = 0;
            for (let i = 0; i < count; i++) {
                const taskData = { ...templateData };
                taskData.mapper_id = mapperId; taskData.mapper_name = mapperName;
                const diffName = count > 1 ? `${mapperName}'s ${baseDiff} ${i + 1}` : `${mapperName}'s ${baseDiff}`;
                taskData.version = diffName;
                const creator = `Mapperatorinator ${model.toUpperCase()}`;
                taskData.display_name = `${artist} - ${title} (${creator}) [${diffName}]`;
                QueueManager.addTask(taskData);
                tasksAdded++;
            }
            this.updateUI();
            Utils.showFlashMessage(`Added ${tasksAdded} task${tasksAdded !== 1 ? 's' : ''} for ${mapperName}`, 'success');
        },

        generateTasksFromMappers() {
            const mappers = MapperManager.getAll();
            if (mappers.length === 0) { Utils.showFlashMessage('No mappers in list!', 'error'); return; }
            const templateData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this), name = $field.attr('name'), type = $field.attr('type');
                if (name && type !== 'file') templateData[name] = type === 'checkbox' ? $field.prop('checked') : ($field.val() || $field.attr('placeholder') || '');
            });
            templateData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) templateData.descriptors.positive.push($cb.val());
                else if ($cb.hasClass('negative-check')) templateData.descriptors.negative.push($cb.val());
            });
            const artist = templateData.artist || '??', title = templateData.title || '??';
            const model = templateData.model || 'v30', stars = parseFloat(templateData.difficulty) || 5.0;
            const baseDiff = DifficultyNameGenerator.getDifficultyName(stars);
            let totalTasks = 0;
            mappers.forEach(mapper => {
                if (!mapper.checked) return;
                const qty = mapper.n || 1;
                for (let i = 0; i < qty; i++) {
                    const taskData = { ...templateData };
                    taskData.mapper_id = mapper.id; taskData.mapper_name = mapper.name;
                    const diffName = qty > 1 ? `${mapper.name}'s ${baseDiff} ${i + 1}` : `${mapper.name}'s ${baseDiff}`;
                    taskData.version = diffName;
                    const creator = `Mapperatorinator ${model.toUpperCase()}`;
                    taskData.display_name = `${artist} - ${title} (${creator}) [${diffName}]`;
                    QueueManager.addTask(taskData);
                    totalTasks++;
                }
            });
            this.updateUI();
            Utils.showFlashMessage(`Generated ${totalTasks} tasks from mapper list!`, 'success');
        },

        updateUI() {
            this.updateQueueList();
            this.updateButtons();
        },

        updateQueueList() {
            const $list = $('#queue-list'), tasks = QueueManager.getAllTasks();
            if (tasks.length === 0) {
                $list.html('<div class="queue-empty">No tasks in queue.</div>');
                $('#queue-count').text('0 tasks'); return;
            }
            let html = '';
            tasks.forEach(task => {
                const displayName = task.formData.display_name || 'Unnamed Task';
                const statusClass = task.status || '';
                const statusIcon = task.status === 'completed' ? '✓ ' : task.status === 'skipped' ? '⏭ ' : '';
                const isRunning = task.status === 'running';
                const removeAction = isRunning ? `QueueUI.skipCurrentTask()` : `QueueUI.removeTask('${task.id}')`;
                html += `<div class="queue-item ${statusClass}" data-task-id="${task.id}">
                    <div class="queue-item-header">
                        <span class="queue-item-name">${statusIcon}${displayName}</span>
                        <button class="queue-item-btn remove" onclick="${removeAction}" title="${isRunning ? 'Skip' : 'Remove'}">×</button>
                    </div>
                </div>`;
            });
            $list.html(html);
            $('#queue-count').text(`${tasks.length} task${tasks.length !== 1 ? 's' : ''}`);
        },

        updateButtons() {
            const queueEmpty = typeof QueueManager !== 'undefined' ? QueueManager.isEmpty() : true;
            const mappersEmpty = typeof MapperManager !== 'undefined' ? MapperManager.getAll().length === 0 : true;
            const hasCheckedMappers = typeof MapperManager !== 'undefined' ? MapperManager.getAll().some(m => m.checked) : false;
            const hasAudio = $('#audio_path').val().trim() !== '';
            const isDetecting = SongDetection.isInProgress();
            const canUseMappers = hasAudio && !isDetecting;

            $('#run-queue-btn').prop('disabled', queueEmpty || this.queueRunning);
            $('#generate-from-mappers-btn').prop('disabled', mappersEmpty || !hasCheckedMappers || !canUseMappers);
            $('.add-single-mapper-btn').prop('disabled', !canUseMappers);

            const $addBtn = $('#add-to-queue-btn');
            if (!hasAudio) $addBtn.prop('disabled', true).attr('title', 'Load an audio file first');
            else if (isDetecting) $addBtn.prop('disabled', true).addClass('detecting').attr('title', 'Detecting...');
            else $addBtn.prop('disabled', false).removeClass('detecting').attr('title', 'Add to queue');
        },

        removeTask(taskId) { QueueManager.removeTask(taskId); this.updateUI(); },
        removeMapper(mapperId) { MapperManager.removeMapper(mapperId); this.updateUI(); },
        incrementMapperQty(mapperId) { MapperManager.updateQuantity(mapperId, 1); this.updateUI(); },
        decrementMapperQty(mapperId) { MapperManager.updateQuantity(mapperId, -1); this.updateUI(); }
    };

    // Make QueueUI globally accessible for onclick handlers
    window.QueueUI = QueueUI;

    // Initialize all components
    function initializeApp() {
        // Initialize language selector
        const $langSelector = $('#language-selector');
        if ($langSelector.length && typeof I18n !== 'undefined') {
            // Set initial value from I18n
            const currentLang = I18n.getCurrentLanguage();
            $langSelector.val(currentLang);

            // Handle language change
            $langSelector.on('change', async function () {
                const newLang = $(this).val();
                const success = await I18n.setLanguage(newLang);
                if (!success) {
                    // Revert to current language if failed
                    $(this).val(I18n.getCurrentLanguage());
                }
            });
        }

        // Check BF16 support on page load
        $.get("/check_bf16_support", function (data) {
            if (data.supported) {
                $("#bf16-option").show();
                if (data.gpu_name) {
                    $("#bf16-gpu-info").text("(" + data.gpu_name + ")");
                }
            }
        });

        // Initialize Select2
        $('.select2').select2({
            placeholder: "Select options",
            allowClear: true,
            dropdownCssClass: "select2-dropdown-dark",
            containerCssClass: "select2-container-dark"
        });

        // Initialize all managers
        FileBrowser.init();
        UIManager.init();
        ValidationManager.init();
        DescriptorManager.init();
        ConfigManager.init();
        InferenceManager.init();

        // Initialize user feature modules
        BeatmapCustomization.init();
        MapperLookup.init();
        DifficultyNameGenerator.init();
        SongDetection.init();
        QueueUI.init();

        // Attach event handlers
        $("#model").on('change', () => UIManager.updateModelSettings());
        $("#gamemode").on('change', () => UIManager.updateConditionalFields());

        // Initial UI updates
        UIManager.updateModelSettings();
    }

    // Start the application
    initializeApp();
});
