import { useState, useEffect, useRef } from 'react';
import './index.css';

interface Dataset {
  key: string;
  name: string;
  description: string;
  modality: string;
  sample_rate: string;
  file_count: number;
  fault_detail: string;
  has_showcase: boolean;
}

interface ComparisonData {
  dataset_name: string;
  fault_detail: string;
  mode: string;
  file_analyzed: string;
  spectrogram_b64: string;
  comparison_b64: string;
  waveform_normal_b64: string;
  waveform_faulty_b64: string;
  waveform_b64: string;
  anomaly_score: number;
  threshold: number;
  score_ratio: number;
  verdict: string;
  peak_freq: string;
  diagnosis: string;
  logs: string[];
  stage2: {
    fault_type: string;
    confidence: number;
    severity: string;
    action: string;
  } | null;
}

const API_BASE = 'http://127.0.0.1:8000';

function PipelineTerminal({ 
  logs, 
  isComplete,
  onTypingComplete 
}: { 
  logs: string[], 
  isComplete: boolean,
  onTypingComplete: () => void 
}) {
  const [displayedIndex, setDisplayedIndex] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // Animate logs appearing one by one without resetting
  useEffect(() => {
    if (displayedIndex < logs.length) {
      const timer = setTimeout(() => {
        setDisplayedIndex(prev => prev + 1);
      }, 150); // Faster typing speed for actual logs
      return () => clearTimeout(timer);
    } else if (isComplete && displayedIndex === logs.length) {
      const timer = setTimeout(() => onTypingComplete(), 600);
      return () => clearTimeout(timer);
    }
  }, [logs.length, displayedIndex, isComplete, onTypingComplete]);

  // Scroll to bottom
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [displayedIndex]);

  const getTime = () => {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
  };

  const colorizeLog = (text: string) => {
    if (!text) return null; // Safeguard against undefined/empty
    if (text.includes("ANOMALY DETECTED")) return <span className="log-danger">{text}</span>;
    if (text.includes("NORMAL")) return <span className="log-success">{text}</span>;
    if (text.includes("STAGE 1") || text.includes("STAGE 2")) return <span style={{ color: '#3b82f6', fontWeight: 600 }}>{text}</span>;
    if (text.includes("Jev-Omni")) return <span style={{ color: '#f59e0b' }}>{text}</span>;
    if (text.includes("->")) return <span style={{ color: '#a78bfa', marginLeft: '1rem' }}>{text}</span>;
    if (text.includes("[Demo Mode]")) return <span style={{ color: '#9ca3af' }}>{text}</span>;
    if (text.includes("==")) return <span style={{ color: '#4b5563' }}>{text}</span>;
    return text;
  };

  return (
    <div className="terminal-section">
      <div className="terminal-header">
        <div className="terminal-dot red"></div>
        <div className="terminal-dot yellow"></div>
        <div className="terminal-dot green"></div>
        <div className="terminal-title">bash — ML Pipeline Diagnostics</div>
      </div>
      <div className="terminal-body">
        {logs.slice(0, displayedIndex).map((log, i) => (
          <div key={i} className="log-line">
            <span className="log-time">[{getTime()}]</span>
            {colorizeLog(log)}
          </div>
        ))}
        {/* Blinking cursor effect at the end if not complete */}
        {!isComplete && (
          <div className="log-line">
            <span className="log-time">[{getTime()}]</span>
            <span style={{ animation: 'fadeIn 1s infinite alternate' }}>_</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  
  // API Response state
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [apiFailed, setApiFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Terminal / Animation State
  const [hasStartedInference, setHasStartedInference] = useState(false);
  const [inferenceLogs, setInferenceLogs] = useState<string[]>([]);
  const [apiIsComplete, setApiIsComplete] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  useEffect(() => {
    // Inject a clean Google Font to override standard sans-serif
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
    document.body.style.fontFamily = "'Outfit', sans-serif";

    fetch(`${API_BASE}/api/datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasets(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch datasets:", err);
        setError("Failed to connect to the API. Is the Flask backend running?");
        setLoading(false);
      });
  }, []);

  const handleCardClick = (datasetKey: string, datasetName: string, modality: string, mode: 'normal' | 'faulty') => {
    setSelectedDataset(datasetKey);
    setComparison(null);
    setApiFailed(false);
    
    setHasStartedInference(true);
    setApiIsComplete(false);
    setShowResults(false);
    
    // Initial connection logs
    setInferenceLogs([
      `[SYSTEM] Invoking Python subprocess: python scripts/inference/predict.py --modality ${modality} --file ${datasetName} --mode ${mode}...`,
      `[SYSTEM] Connecting to backend engine...`,
    ]);

    fetch(`${API_BASE}/api/analyze/${datasetKey}?mode=${mode}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch comparison data");
        return res.json();
      })
      .then((data: ComparisonData) => {
        setComparison(data);
        // Feed the backend-generated logs into the terminal!
        setInferenceLogs(prev => [
          ...prev, 
          ...data.logs
        ]);
        setApiIsComplete(true);
      })
      .catch(err => {
        console.error("Failed to fetch comparison:", err);
        setApiFailed(true);
        setInferenceLogs(prev => [
          ...prev, 
          `[ERROR] Subprocess failed: ${err.message}`
        ]);
        setApiIsComplete(true);
      });
  };

  const handleTerminalDoneTyping = () => {
    if (comparison && !apiFailed) {
      setShowResults(true);
    }
  };

  const closeResults = () => {
    setSelectedDataset(null);
    setHasStartedInference(false);
    setShowResults(false);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Canary</h1>
        <p>Zero-Shot Multimodal Fault Detection System</p>
      </header>

      {error && (
        <div style={{ backgroundColor: '#fef2f2', padding: '1rem', borderRadius: '8px', color: '#b91c1c', border: '1px solid #f87171', marginBottom: '2rem' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="loader">
          <div className="spinner"></div>
          <p>Initializing System...</p>
        </div>
      ) : (
        <div className="main-content">
          <div className="sidebar">
            {datasets.map(ds => (
              <div 
                key={ds.key} 
                className="card" 
                onMouseEnter={() => setHoveredCard(ds.key)}
                onMouseLeave={() => setHoveredCard(null)}
              >
                <div className="card-header">
                  <h3 className="card-title">{ds.name}</h3>
                  <span className="badge-modality">{ds.modality}</span>
                </div>
                <p className="card-desc">{ds.description}</p>
                <div className="card-meta">
                  <span className="meta-item">
                    📁 {ds.file_count.toLocaleString()} files
                  </span>
                  <span className="meta-item">
                    ⏱️ {ds.sample_rate}
                  </span>
                </div>
                {hoveredCard === ds.key && (
                  <div className="card-hover-actions">
                    <button 
                      className="hover-btn hover-btn-normal"
                      onClick={(e) => { e.stopPropagation(); handleCardClick(ds.key, ds.name, ds.modality, 'normal'); }}
                    >
                      Analyze Normal
                    </button>
                    <button 
                      className="hover-btn hover-btn-faulty"
                      onClick={(e) => { e.stopPropagation(); handleCardClick(ds.key, ds.name, ds.modality, 'faulty'); }}
                    >
                      Analyze Anomaly
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {/* Right Panel Logic */}
          {!hasStartedInference ? (
            <div style={{ flex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f1f5f9', borderRadius: '12px', border: '2px dashed #cbd5e1', height: 'calc(100vh - 150px)', color: '#64748b', position: 'sticky', top: '2rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>⚗️</div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem' }}>Pipeline Idle</h2>
              <p>Select a dataset card on the left to execute the inference script.</p>
            </div>
          ) : !showResults ? (
            <PipelineTerminal 
              logs={inferenceLogs} 
              isComplete={apiIsComplete} 
              onTypingComplete={handleTerminalDoneTyping} 
            />
          ) : comparison ? (
            <div className="results-panel slide-in-right">
              <div className="results-header">
                <h2>Diagnostic Results: {comparison.dataset_name}</h2>
                <button className="close-btn" onClick={closeResults}>&times;</button>
              </div>
              
              <div className="comparison-container">
                
                {/* Stage 1: Edge Analysis Report */}
                <div className="analysis-report">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 className="analysis-header" style={{ color: 'var(--text-primary)', fontWeight: 700, margin: 0 }}>
                      Stage 1: Edge PCAReconstructionMemoryBank
                    </h3>
                    <div style={{ padding: '0.25rem 0.75rem', borderRadius: '999px', fontSize: '0.85rem', fontWeight: 700, backgroundColor: comparison.verdict === 'NORMAL' ? '#dcfce7' : '#fee2e2', color: comparison.verdict === 'NORMAL' ? '#166534' : '#991b1b' }}>
                      {comparison.verdict.replace('_', ' ')}
                    </div>
                  </div>
                  <div className="analysis-grid">
                    <div className="analysis-metric">
                      <div className="metric-label">Calculated Anomaly Score</div>
                      <div className="metric-value danger">{comparison.anomaly_score.toFixed(4)}</div>
                      <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '0.5rem' }}>Threshold limit: {comparison.threshold.toFixed(4)}</div>
                    </div>
                    <div className="analysis-metric">
                      <div className="metric-label">Deviation Ratio</div>
                      <div className="metric-value">{comparison.score_ratio.toFixed(2)}x</div>
                      <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '0.5rem' }}>Multiplier over baseline</div>
                    </div>
                    <div className="analysis-metric">
                      <div className="metric-label">Peak Frequency Deviation</div>
                      <div className="metric-value">{comparison.peak_freq}</div>
                    </div>
                  </div>
                </div>

                {/* Stage 2: Cloud Analysis Report */}
                {comparison.stage2 && (
                  <div className="analysis-report" style={{ backgroundColor: '#fffbeb', borderColor: '#fde68a', marginTop: '1.5rem' }}>
                    <h3 className="analysis-header" style={{ color: '#92400e', marginBottom: '1.5rem', fontWeight: 700 }}>
                      Stage 2: TypeSafe AI Jev-Omni Cloud Classification
                    </h3>
                    <div className="analysis-grid">
                      <div className="analysis-metric" style={{ backgroundColor: 'white' }}>
                        <div className="metric-label">Classified Fault Type</div>
                        <div className="metric-value" style={{ color: '#d97706' }}>{comparison.stage2.fault_type}</div>
                        <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '0.5rem' }}>Confidence: {(comparison.stage2.confidence * 100).toFixed(1)}%</div>
                      </div>
                      <div className="analysis-metric" style={{ backgroundColor: 'white' }}>
                        <div className="metric-label">Severity</div>
                        <div className="metric-value" style={{ color: '#b91c1c' }}>{comparison.stage2.severity}</div>
                      </div>
                      <div className="analysis-metric" style={{ backgroundColor: 'white' }}>
                        <div className="metric-label">Recommended Action</div>
                        <div className="metric-value" style={{ fontSize: '1.1rem' }}>{comparison.stage2.action}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Architecture Specs */}
                <div className="architecture-report" style={{ marginTop: '1.5rem', padding: '1.5rem', backgroundColor: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                  <h4 style={{ color: '#475569', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '1rem', fontWeight: 700 }}>System Architecture Specs</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', fontSize: '0.9rem', color: '#64748b' }}>
                    <div><strong>Encoder:</strong> SpectrogramEncoder (4-layer CNN, DANN)</div>
                    <div><strong>Memory Bank:</strong> PCA Reconstruction (32 components, StandardScaler)</div>
                    <div><strong>Input Config:</strong> {comparison.modality === 'vibration' ? '25600Hz' : '16000Hz'}, n_mels=128, n_fft=1024</div>
                    <div><strong>Training Constraint:</strong> Normal-only, Zero-shot Fault Detection</div>
                  </div>
                </div>

                {/* Heatmap comparison panel */}
                {comparison.verdict === 'ANOMALY_DETECTED' ? (
                  <>
                    <div style={{ marginTop: '2rem' }}>
                      <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: 'var(--text-primary)', fontWeight: 700 }}>
                        Spectrogram Difference Matrix
                      </h3>
                      <img 
                        src={`data:image/png;base64,${comparison.comparison_b64}`} 
                        alt="Comparison Heatmap" 
                        className="comparison-image"
                      />
                    </div>
                    
                    {/* Waveforms side-by-side */}
                    <div className="waveforms-grid">
                      <div className="waveform-card normal">
                        <h4 style={{ fontWeight: 600, color: '#334155' }}>Baseline Signal Profile</h4>
                        <img src={`data:image/png;base64,${comparison.waveform_normal_b64}`} alt="Normal Waveform" />
                      </div>
                      
                      <div className="waveform-card faulty">
                        <h4 style={{ fontWeight: 600, color: 'var(--danger-color)' }}>Anomalous Signal Profile</h4>
                        <img src={`data:image/png;base64,${comparison.waveform_faulty_b64}`} alt="Faulty Waveform" />
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ marginTop: '2rem' }}>
                      <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: 'var(--text-primary)', fontWeight: 700 }}>
                        Spectrogram Profile
                      </h3>
                      <img 
                        src={`data:image/png;base64,${comparison.spectrogram_b64}`} 
                        alt="Normal Spectrogram" 
                        className="comparison-image"
                        style={{ maxWidth: '600px', margin: '0 auto', display: 'block' }}
                      />
                    </div>
                    
                    <div style={{ marginTop: '2rem' }}>
                      <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: 'var(--text-primary)', fontWeight: 700 }}>
                        Signal Waveform
                      </h3>
                      <img 
                        src={`data:image/png;base64,${comparison.waveform_b64}`} 
                        alt="Normal Waveform" 
                        className="comparison-image"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="results-panel">
              <div style={{ color: '#dc2626', textAlign: 'center', padding: '2rem' }}>
                Failed to load diagnostic data. Check backend logs.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
