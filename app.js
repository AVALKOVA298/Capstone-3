// app.js
class JobFakeDetector {
    constructor() {
        this.model = null;              // tf.GraphModel
        this.maxLen = 0;
        this.wordIndex = {};
        this.isModelLoaded = false;
        this.isDataLoaded = false;
        this.edaData = null;

        this.init();
    }

    async init() {
        try {
            await Promise.all([
                this.loadModel(),
                this.loadTokenizer(),
                this.loadEdaData()
            ]);

            this.isModelLoaded = true;
            this.isDataLoaded = true;

            this.setupEventListeners();
            this.renderEda();
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to load model or data. Please refresh the page.');
        }
    }

    // ---------- MODEL & TOKENIZER ----------

    async loadModel() {
        try {
            // graph-model с сигнатурой keras_tensor_11 -> Identity:0
            this.model = await tf.loadGraphModel('model/model.json');
            console.log('Model loaded successfully (graph model)');
            console.log('Inputs:', this.model.inputs);
            console.log('Outputs:', this.model.outputs);
        } catch (error) {
            throw new Error(`Model loading failed: ${error.message}`);
        }
    }

    async loadTokenizer() {
        try {
            const response = await fetch('model/frontend_config.json');
            const config = await response.json();
            this.maxLen = config.max_len;
            this.wordIndex = config.word_index || {};
            console.log('Tokenizer loaded:', Object.keys(this.wordIndex).length, 'words');
        } catch (error) {
            throw new Error(`Tokenizer loading failed: ${error.message}`);
        }
    }

    async loadEdaData() {
        try {
            const edaResponse = await fetch('data/eda_data.json');
            if (edaResponse.ok) {
                this.edaData = await edaResponse.json();
                return;
            }

            await this.processCsvData();
        } catch (error) {
            console.warn('EDA data loading failed, using fallback:', error);
            this.edaData = this.getMockEdaData();
        }
    }

    async processCsvData() {
        const zipResponse = await fetch('data/combined_dataset_clean.zip');
        const zipArrayBuffer = await zipResponse.arrayBuffer();
        const zip = new JSZip();
        await zip.loadAsync(zipArrayBuffer);

        const csvFile = Object.values(zip.files).find(f => f.name.endsWith('.csv'));
        if (!csvFile) throw new Error('CSV not found in ZIP');

        const csvContent = await csvFile.async('string');
        const parsed = Papa.parse(csvContent, { header: true, skipEmptyLines: true });
        this.edaData = this.computeEdaStats(parsed.data);
    }

    computeEdaStats(data) {
        const realJobs = data.filter(row => row.fraudulent === '0');
        const fakeJobs = data.filter(row => row.fraudulent === '1');

        const textLengths = data.map(row => {
            const text = [
                row.title,
                row.description,
                row.company_profile,
                row.requirements,
                row.benefits
            ].filter(Boolean).join(' ').length;

            return { length: text, isFake: row.fraudulent === '1' };
        });

        return {
            classDistribution: {
                real: realJobs.length,
                fake: fakeJobs.length,
                total: data.length
            },
            textLengths,
            missingValues: this.computeMissingValues(data),
            realWords: this.extractWords(realJobs),
            fakeWords: this.extractWords(fakeJobs)
        };
    }

    computeMissingValues(data) {
        const fields = ['title', 'description', 'company_profile', 'requirements', 'benefits'];
        const stats = { real: {}, fake: {} };

        data.forEach(row => {
            const isFake = row.fraudulent === '1';
            const group = isFake ? 'fake' : 'real';

            fields.forEach(field => {
                if (!stats[group][field]) stats[group][field] = 0;
                if (!row[field] || row[field].trim() === '') {
                    stats[group][field]++;
                }
            });
        });

        return stats;
    }

    extractWords(jobs) {
        const wordFreq = {};
        jobs.forEach(job => {
            const text = [job.title, job.description, job.company_profile]
                .filter(Boolean)
                .join(' ')
                .toLowerCase();

            const words = text.match(/\b\w+\b/g) || [];
            words.forEach(word => {
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            });
        });

        return Object.entries(wordFreq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 100);
    }

    getMockEdaData() {
        return {
            classDistribution: { real: 14856, fake: 866, total: 15722 },
            textLengths: Array(100).fill(0).map((_, i) => ({
                length: 200 + i * 20 + Math.random() * 100,
                isFake: i % 10 === 0
            })),
            missingValues: {
                real: { title: 120, description: 45, company_profile: 800, requirements: 200, benefits: 300 },
                fake: { title: 50, description: 20, company_profile: 150, requirements: 80, benefits: 100 }
            },
            realWords: [['job', 5000], ['experience', 3500], ['team', 2800], ['work', 2500], ['skills', 2200]],
            fakeWords: [['money', 800], ['urgent', 650], ['immediate', 550], ['apply', 500], ['now', 450]]
        };
    }

    // ---------- UI ----------

    setupEventListeners() {
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.target.dataset.tab;
                this.switchTab(tab);
            });
        });

        const form = document.getElementById('job-form');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.predictJob();
            });
        }

        const predictBtn = document.getElementById('predict-btn');
        if (predictBtn) {
            predictBtn.disabled = !this.isModelLoaded;
        }
    }

    switchTab(activeTab) {
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        const btn = document.querySelector(`[data-tab="${activeTab}"]`);
        const tab = document.getElementById(activeTab);

        if (btn) btn.classList.add('active');
        if (tab) tab.classList.add('active');
    }

    showError(message) {
        const active = document.querySelector('.tab-content.active');
        if (!active) return;
        active.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #ef4444;">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }

    // ---------- TEXT PREPROCESSING ----------

    getFullText() {
        const fields = [
            'title',
            'company_profile',
            'description',
            'requirements',
            'benefits',
            'location',
            'salary_range',
            'employment_type',
            'industry'
        ];

        return fields
            .map(id => {
                const el = document.getElementById(id);
                return el ? el.value.trim() : '';
            })
            .filter(text => text)
            .join(' ');
    }

    tokenizeText(text) {
        const lowerText = text.toLowerCase();
        const words = lowerText.match(/\b\w+\b/g) || [];

        const sequence = words.map(word => this.wordIndex[word] || 1); // 1 = OOV

        // post‑padding до maxLen
        const padded = sequence.concat(Array(Math.max(this.maxLen - sequence.length, 0)).fill(0));
        const inputArray = padded.slice(0, this.maxLen);

        // tensor [1, max_len] из float32 (ожидается keras_tensor_11:0, DT_FLOAT, [-1,300])
        // если maxLen > 300, обрежем до 300, если < 300 — допадим нулями
        const fixedLen = 300;
        let data = inputArray;
        if (data.length > fixedLen) {
            data = data.slice(0, fixedLen);
        } else if (data.length < fixedLen) {
            data = data.concat(Array(fixedLen - data.length).fill(0));
        }

        return tf.tensor2d([data], [1, fixedLen], 'float32');
    }

    // ---------- INFERENCE С GRAPH MODEL ----------

    async predictJob() {
        const fullText = this.getFullText();
        if (!fullText.trim()) {
            alert('Please fill at least one field');
            return;
        }

        const predictBtn = document.getElementById('predict-btn');
        const originalText = predictBtn.textContent;
        predictBtn.textContent = 'Analyzing...';
        predictBtn.disabled = true;

        try {
            const inputTensor = this.tokenizeText(fullText);

            // имя входа и выхода берём из signature в model.json: keras_tensor_11 -> Identity:0 [file:93]
            const inputName = 'keras_tensor_11';
            const outputName = 'Identity:0';

            const outputTensor = await this.model.executeAsync(
                { [inputName]: inputTensor },
                outputName
            );

            const probArr = await outputTensor.data();
            const prob = probArr[0];

            this.displayPrediction(prob);

            inputTensor.dispose();
            outputTensor.dispose();
        } catch (error) {
            console.error('Prediction failed:', error);
            alert('Prediction failed. Please check the console for details.');
        } finally {
            predictBtn.textContent = originalText;
            predictBtn.disabled = false;
        }
    }

    displayPrediction(prob) {
    const resultEl = document.getElementById('prediction-result');
    const labelEl = document.getElementById('prediction-label');
    const probEl = document.getElementById('probability-value');
    const barEl = document.getElementById('prob-bar');

    if (!resultEl || !labelEl || !probEl || !barEl) return;

    resultEl.style.display = 'block';

    const isFake = prob >= 0.5;
    const label = isFake ? '⚠️ SUSPICIOUS' : '✅ LEGIT';
    const colorClass = isFake ? 'suspicious' : 'real';

    labelEl.textContent = label;
    labelEl.className = colorClass;
    probEl.textContent = `${(prob * 100).toFixed(1)}% fake probability`;

    const barWidth = Math.min(prob * 100, 100);
    barEl.style.width = `${barWidth}%`;
}

