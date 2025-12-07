// app.js
class JobFakeDetector {
    constructor() {
        this.model = null;
        this.tokenizerConfig = null;
        this.maxLen = 0;
        this.wordIndex = {};
        this.isModelLoaded = false;
        this.isDataLoaded = false;
        
        this.init();
    }

    async init() {
        this.showLoading('job-check', 'Loading model...');
        this.showLoading('eda', 'Loading EDA data...');
        
        try {
            // Load model and tokenizer in parallel
            await Promise.all([
                this.loadModel(),
                this.loadTokenizer(),
                this.loadEdaData()
            ]);
            
            this.isModelLoaded = true;
            this.isDataLoaded = true;
            this.hideLoading();
            this.setupEventListeners();
            this.renderEda();
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to load model or data. Please refresh the page.');
        }
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('model/model.json');
            console.log('Model loaded successfully');
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
            // Try precomputed data first
            const edaResponse = await fetch('data/eda_data.json');
            if (edaResponse.ok) {
                this.edaData = await edaResponse.json();
                return;
            }
            
            // Fallback to CSV processing
            await this.processCsvData();
        } catch (error) {
            console.warn('EDA data loading failed:', error);
            // Use mock data as fallback
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
            const text = [row.title, row.description, row.company_profile, row.requirements, row.benefits]
                .filter(Boolean).join(' ').length;
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
            const text = [job.title, job.description, job.company_profile].join(' ').toLowerCase();
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

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.target.dataset.tab;
                this.switchTab(tab);
            });
        });

        // Form submission
        document.getElementById('job-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.predictJob();
        });

        // Enable predict button when model is ready
        document.getElementById('predict-btn').disabled = false;
    }

    switchTab(activeTab) {
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        document.querySelector(`[data-tab="${activeTab}"]`).classList.add('active');
        document.getElementById(activeTab).classList.add('active');
    }

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
            const prediction = this.model.predict(inputTensor);
            const prob = await prediction.data();
            
            this.displayPrediction(Array.from(prob)[0]);
            inputTensor.dispose();
            prediction.dispose();
        } catch (error) {
            console.error('Prediction failed:', error);
            alert('Prediction failed. Please try again.');
        } finally {
            predictBtn.textContent = originalText;
            predictBtn.disabled = false;
        }
    }

    getFullText() {
        const fields = ['title', 'company_profile', 'description', 'requirements', 'benefits', 
                       'location', 'salary_range', 'employment_type', 'industry'];
        return fields.map(id => {
            const el = document.getElementById(id);
            return el.value.trim();
        }).filter(text => text).join(' ');
    }

    tokenizeText(text) {
        const lowerText = text.toLowerCase();
        const words = lowerText.match(/\b\w+\b/g) || [];
        const sequence = words.map(word => this.wordIndex[word] || 1); // 1 = OOV
        const padded = sequence.concat(Array(this.maxLen - sequence.length).fill(0));
        const inputArray = padded.slice(0, this.maxLen);
        
        return tf.tensor2d([inputArray], [1, this.maxLen]);
    }

    displayPrediction(prob) {
        const resultEl = document.getElementById('prediction-result');
        const labelEl = document.getElementById('prediction-label');
        const probEl = document.getElementById('probability-value');
        const barEl = document.getElementById('prob-bar');
        
        resultEl.style.display = 'block';
        
        const isFake = prob >= 0.5;
        const label = isFake ? '⚠️ SUSPICIOUS' : '✅ LEGIT';
        const colorClass = isFake ? 'suspicious' : 'real';
        
        labelEl.textContent = label;
        labelEl.className = `prediction-label ${colorClass}`;
        probEl.textContent = `${(prob * 100).toFixed(1)}% fake probability`;
        
        const barWidth = Math.min(prob * 100, 100);
        barEl.style.width = `${barWidth}%`;
    }

    renderEda() {
        if (!this.edaData) return;
        
        this.renderClassDistribution();
        this.renderBoxplot();
        this.renderMissingValues();
        this.renderWordClouds();
    }

    renderClassDistribution() {
        const ctx = document.getElementById('class-dist-chart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Real Jobs', 'Fake Jobs'],
                datasets: [{
                    data: [this.edaData.classDistribution.real, this.edaData.classDistribution.fake],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    renderBoxplot() {
        const ctx = document.getElementById('boxplot-chart').getContext('2d');
        const realLengths = this.edaData.textLengths
            .filter(d => !d.isFake).map(d => d.length);
        const fakeLengths = this.edaData.textLengths
            .filter(d => d.isFake).map(d => d.length);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Real Jobs', 'Fake Jobs'],
                datasets: [{
                    label: 'Text Length Distribution',
                    data: [
                        { y: this.stats(realLengths) },
                        { y: this.stats(fakeLengths) }
                    ],
                    backgroundColor: ['rgba(16, 185, 129, 0.3)', 'rgba(239, 68, 68, 0.3)'],
                    borderColor: ['#10b981', '#ef4444'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Characters' } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const stats = ctx.raw.y;
                                return `Q1: ${stats.q1}, Median: ${stats.median}, Q3: ${stats.q3}`;
                            }
                        }
                    }
                }
            }
        });
    }

    stats(arr) {
        const sorted = [...arr].sort((a, b) => a - b);
        const n = sorted.length;
        return {
            min: Math.min(...arr),
            q1: sorted[Math.floor(n * 0.25)],
            median: sorted[Math.floor(n * 0.5)],
            q3: sorted[Math.floor(n * 0.75)],
            max: Math.max(...arr)
        };
    }

    renderMissingValues() {
        const ctx = document.getElementById('missing-values-chart').getContext('2d');
        const fields = ['title', 'description', 'company_profile', 'requirements', 'benefits'];
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: fields,
                datasets: [
                    {
                        label: 'Real Jobs',
                        data: fields.map(f => this.edaData.missingValues.real[f] || 0),
                        backgroundColor: 'rgba(16, 185, 129, 0.6)'
                    },
                    {
                        label: 'Fake Jobs',
                        data: fields.map(f => this.edaData.missingValues.fake[f] || 0),
                        backgroundColor: 'rgba(239, 68, 68, 0.6)'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, beginAtZero: true }
                }
            }
        });
    }

    renderWordClouds() {
        this.renderWordCloud('real-wordcloud', this.edaData.realWords, '#10b981');
        this.renderWordCloud('fake-wordcloud', this.edaData.fakeWords, '#ef4444');
    }

    renderWordCloud(elementId, words, color) {
        const svg = d3.select(`#${elementId}`);
        const width = 400, height = 300;
        
        d3.layout.cloud()
            .size([width, height])
            .words(words.map(([text, freq]) => ({text, size: Math.sqrt(freq) * 3})))
            .padding(5)
            .rotate(() => ~~(Math.random() * 2) * 90)
            .font("Impact")
            .fontSize(d => d.size)
            .on("end", draw)
            .start();

        function draw(words) {
            svg.append("g")
                .attr("transform", "translate(" + width/2 + "," + height/4 + ")")
                .selectAll("text")
                .data(words)
                .enter().append("text")
                .style("font-size", d => d.size + "px")
                .style("fill", color)
                .style("font-weight", "bold")
                .attr("text-anchor", "middle")
                .attr("transform", d => `translate(${d.x}, ${d.y}) rotate(${d.rotate})`)
                .text(d => d.text);
        }
    }

    showLoading(tabId, message) {
        const tab = document.getElementById(tabId);
        tab.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>${message}</p>
            </div>
        `;
    }

    hideLoading() {
        // Clear loading states when data is ready
    }

    showError(message) {
        document.querySelector('.tab-content.active').innerHTML = `
            <div style="text-align: center; padding: 40px; color: #ef4444;">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new JobFakeDetector();
});
