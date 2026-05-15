import { useCallback, useEffect, useRef, useState } from "react";
import Dashboard from "./pages/Dashboard";
import Glossary from "./pages/Glossary";
import History from "./pages/History";
import Results from "./pages/Results";
import Upload from "./pages/Upload";
import GlossaryModal from "./components/GlossaryModal";
import { APP_PAGES, getConceptInterpretation, getConceptMeta, resolveConceptKey } from "./utils/constants";
import {
  clearSessions,
  deleteSession,
  loadSessions,
  loadTheme,
  saveTheme,
  upsertSession,
} from "./utils/storage";

const PAGE_TITLES = {
  dashboard: "Dashboard",
  upload: "New Analysis",
  results: "Results",
  history: "History",
  glossary: "Glossary",
};

const STAGE_LABELS = ["Preparing", "Reviewing", "Summarizing", "Finalizing"];

function createInitialStages() {
  return STAGE_LABELS.map((label, index) => ({
    label,
    index,
    status: "pending",
    startedAt: null,
    completedAt: null,
  }));
}

function createInitialRunState() {
  return {
    loading: false,
    error: "",
    progress: 0,
    stages: createInitialStages(),
  };
}

function getRouteFromHash() {
  const hash = window.location.hash.replace("#", "");
  return PAGE_TITLES[hash] ? hash : "dashboard";
}

function buildApiBase(baseUrl) {
  const trimmed = (baseUrl || "http://localhost:8000").replace(/\/$/, "");
  return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
}

function getStageIndex(progress) {
  if (progress < 25) {
    return 0;
  }
  if (progress < 55) {
    return 1;
  }
  if (progress < 85) {
    return 2;
  }
  return 3;
}

function parseReportSections(report) {
  if (!report) {
    return null;
  }
  if (typeof report === "string") {
    return { raw_report_text: report };
  }
  const clinicalInterpretation = [
    report.key_findings || "",
    report.clinical_interpretation || "",
  ].filter(Boolean).join("\n\n");

  return {
    patient_recording_information: report.patient_recording_information || "",
    model_prediction_summary: report.model_prediction_summary || "",
    concept_based_explanation: report.concept_based_explanation || "",
    clinical_interpretation: clinicalInterpretation,
    confidence_reliability_assessment: report.confidence_reliability_assessment || "",
    safety_disclaimer: report.safety_disclaimer || "",
    raw_report_text: report.raw_report_text || "",
  };
}

function normalizeProbability(value) {
  const numeric = Number(value || 0);
  if (numeric > 1) {
    return numeric / 100;
  }
  return numeric;
}

function normalizeConfidence(value) {
  const numeric = Number(value || 0);
  return numeric <= 1 ? numeric * 100 : numeric;
}

function formatReportText(reportSections) {
  if (!reportSections) {
    return "";
  }
  return [
    `Patient & Recording Information\n${reportSections.patient_recording_information}`,
    `Clinical Summary\n${reportSections.model_prediction_summary}`,
    `Biomarker Interpretation\n${reportSections.concept_based_explanation}`,
    `Clinical Interpretation\n${reportSections.clinical_interpretation}`,
    `Confidence & Reliability Assessment\n${reportSections.confidence_reliability_assessment}`,
    `Safety Disclaimer\n${reportSections.safety_disclaimer}`,
    reportSections.raw_report_text ? `Raw Report Text\n${reportSections.raw_report_text}` : "",
  ].join("\n\n");
}

function normalizeConcepts(result, predictionLabel) {
  const explanationConcepts = result.raw_explanation?.concepts || result.concepts || {};
  const tcavConcepts = Array.isArray(result.tcav_concepts) ? result.tcav_concepts : [];

  if (Array.isArray(explanationConcepts)) {
    return explanationConcepts.map((concept) => ({
      name: resolveConceptKey(concept.concept_name || concept.name),
      tcavScore: Number(concept.tcav_score || 0),
      meanDd: Number(concept.mean_derivative || concept.mean_dd || 0),
      stdDd: Number(concept.std_dd || concept.tcav_std || 0),
      cavAccuracy: normalizeConfidence(concept.cav_accuracy || 0),
      clinicalFlag: concept.clinical_flag || "WEAK",
      segmentDd: concept.segment_dd || [],
      predictionLabel,
    }));
  }

  const names = Object.keys(explanationConcepts);
  if (names.length > 0) {
    return names.map((name) => {
      const source = explanationConcepts[name];
      return {
        name: resolveConceptKey(name),
        tcavScore: Number(source.tcav_score || 0),
        meanDd: Number(source.mean_dd || 0),
        stdDd: Number(source.std_dd || 0),
        cavAccuracy: normalizeConfidence(source.cav_accuracy || 0),
        clinicalFlag: source.clinical_flag || "WEAK",
        segmentDd: Array.isArray(source.segment_dd) ? source.segment_dd : [],
        predictionLabel,
      };
    });
  }

  return tcavConcepts.map((concept) => ({
    name: resolveConceptKey(concept.concept_name || concept.name),
    tcavScore: Number(concept.tcav_score || 0),
    meanDd: Number(concept.mean_derivative || concept.mean_dd || 0),
    stdDd: Number(concept.std_dd || concept.tcav_std || 0),
    cavAccuracy: normalizeConfidence(concept.cav_accuracy || 0),
    clinicalFlag: concept.clinical_flag || "WEAK",
    segmentDd: Array.isArray(concept.segment_dd) ? concept.segment_dd : [],
    predictionLabel,
  }));
}

function normalizeAnalysis(result, report, subjectFallback, jobId = null) {
  const explanation = result.raw_explanation || result;
  const predictionLabel = result.prediction?.label || result.prediction || "Healthy";
  const probabilities = result.prediction?.probabilities || [];
  const mddProb = normalizeProbability(explanation.mdd_prob || result.mdd_prob || probabilities[1] || 0);
  const hcProb = normalizeProbability(explanation.hc_prob || result.hc_prob || probabilities[0] || 0);
  const confidence = normalizeConfidence(explanation.confidence || result.confidence || result.prediction?.confidence || 0);
  const nSegments = Number(explanation.n_segments || result.n_segments || explanation.segment_mdd_probs?.length || 0);
  const subject = result.subject_id || explanation.subject || subjectFallback || "Unknown subject";
  const reportSections = parseReportSections(report);
  const concepts = normalizeConcepts(result, predictionLabel).map((concept) => ({
    ...concept,
    meta: getConceptMeta(concept.name),
    interpretation: getConceptInterpretation(concept),
  }));

  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    jobId,
    subject,
    createdAt: new Date().toISOString(),
    prediction: predictionLabel,
    confidence,
    mddProb,
    hcProb,
    nSegments,
    recordingSeconds: Number(explanation.recording_s || result.recording_s || nSegments * 5),
    segmentProbabilities: explanation.segment_mdd_probs || result.segment_mdd_probs || [],
    concepts,
    reportSections,
    reportText: result.report_text || formatReportText(reportSections),
    rawResult: result,
  };
}

function updateStageStates(previousStages, progress) {
  const now = Date.now();
  const activeIndex = getStageIndex(progress);

  return previousStages.map((stage, index) => {
    const next = { ...stage };
    if (index < activeIndex || progress >= 100) {
      next.status = "complete";
      next.startedAt = next.startedAt || now;
      next.completedAt = next.completedAt || now;
      return next;
    }
    if (index === activeIndex) {
      next.status = "active";
      next.startedAt = next.startedAt || now;
      if (progress >= 100) {
        next.status = "complete";
        next.completedAt = next.completedAt || now;
      }
      return next;
    }
    return next;
  });
}

export default function App() {
  const [route, setRoute] = useState(getRouteFromHash());
  const [theme, setTheme] = useState(loadTheme());
  const [sessions, setSessions] = useState(loadSessions());
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [healthStatus, setHealthStatus] = useState("checking");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(window.innerWidth < 900);
  const [runState, setRunState] = useState(createInitialRunState());
  const [glossaryModal, setGlossaryModal] = useState(null);
  const streamRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    saveTheme(theme);
  }, [theme]);

  useEffect(() => {
    const onHashChange = () => setRoute(getRouteFromHash());
    const onResize = () => {
      if (window.innerWidth < 900) {
        setSidebarCollapsed(true);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  useEffect(() => {
    if (route === "results" && !currentAnalysis && sessions.length > 0) {
      setCurrentAnalysis(sessions[0]);
    }
  }, [currentAnalysis, route, sessions]);

  useEffect(() => {
    let active = true;

    async function ping() {
      try {
        const response = await fetch(`${buildApiBase("http://localhost:8000")}/health`);
        const data = await response.json();
        if (active) {
          setHealthStatus(data.status === "ok" ? "online" : "offline");
        }
      } catch (error) {
        if (active) {
          setHealthStatus("offline");
        }
      }
    }

    ping();
    const timer = window.setInterval(ping, 10000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const navigate = useCallback((nextRoute) => {
    window.location.hash = nextRoute;
    setRoute(nextRoute);
  }, []);

  const handleThemeToggle = useCallback(() => {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  }, []);

  const handleDeleteSession = useCallback((sessionId) => {
    const next = deleteSession(sessionId);
    setSessions(next);
    if (currentAnalysis && currentAnalysis.id === sessionId) {
      setCurrentAnalysis(null);
    }
  }, [currentAnalysis]);

  const handleClearAll = useCallback(() => {
    setSessions(clearSessions());
    setCurrentAnalysis(null);
  }, []);

  const handleViewSession = useCallback((sessionId) => {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) {
      return;
    }
    setCurrentAnalysis(session);
    navigate("results");
  }, [navigate, sessions]);

  const handleResetRun = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setRunState(createInitialRunState());
  }, []);

  const handleStartAnalysis = useCallback(async (file, subjectName) => {
    handleResetRun();
    setRunState((prev) => ({
      ...prev,
      loading: true,
      progress: 1,
      stages: updateStageStates(createInitialStages(), 1),
    }));
    navigate("upload");

    const apiBase = buildApiBase("http://localhost:8000");
    const form = new FormData();
    form.append("file", file);
    form.append("subject_id", subjectName);

    try {
      const response = await fetch(`${apiBase}/analyze`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      if (data.job_id) {
        const stream = new EventSource(`${apiBase}/stream/${data.job_id}`);
        streamRef.current = stream;

        stream.onmessage = (event) => {
          const payload = JSON.parse(event.data);
          if (payload.message === "done") {
            stream.close();
            streamRef.current = null;
            const normalized = normalizeAnalysis(payload.result, payload.report, subjectName, data.job_id);
            const saved = upsertSession(normalized);
            setSessions(saved);
            setCurrentAnalysis(normalized);
          setRunState({
            loading: false,
            error: "",
            progress: 100,
            stages: updateStageStates(createInitialStages(), 100),
          });
            navigate("results");
            return;
          }

          if (typeof payload.message === "string" && payload.message.startsWith("error:")) {
            stream.close();
            streamRef.current = null;
            setRunState((prev) => ({
              ...prev,
              loading: false,
              error: payload.message,
            }));
            return;
          }

          setRunState((prev) => ({
            ...prev,
            progress: payload.progress || prev.progress,
            stages: updateStageStates(prev.stages, payload.progress || 0),
          }));
        };

        stream.onerror = () => {
          stream.close();
          streamRef.current = null;
          setRunState((prev) => ({
            ...prev,
            loading: false,
            error: "Connection lost while streaming analysis progress.",
          }));
        };
        return;
      }

      const normalized = normalizeAnalysis(data, null, subjectName, data.job_id || null);
      const saved = upsertSession(normalized);
      setSessions(saved);
      setCurrentAnalysis(normalized);
      setRunState({
        loading: false,
        error: "",
        progress: 100,
        stages: updateStageStates(createInitialStages(), 100),
      });
      navigate("results");
    } catch (error) {
      setRunState((prev) => ({
        ...prev,
        loading: false,
        error: error.message,
      }));
    }
  }, [handleResetRun, navigate]);

  const renderPage = () => {
    if (route === "dashboard") {
      return (
        <Dashboard
          sessions={sessions}
          onStartNew={() => navigate("upload")}
          onViewSession={handleViewSession}
          onDeleteSession={handleDeleteSession}
        />
      );
    }
    if (route === "upload") {
      return (
        <Upload
          onAnalyze={handleStartAnalysis}
          runState={runState}
          onRetry={handleResetRun}
        />
      );
    }
    if (route === "results") {
      return <Results analysis={currentAnalysis} />;
    }
    if (route === "history") {
      return (
        <History
          sessions={sessions}
          onViewSession={handleViewSession}
          onDeleteSession={handleDeleteSession}
          onClearAll={handleClearAll}
        />
      );
    }
    if (route === "glossary") {
      return <Glossary onOpenConcept={setGlossaryModal} />;
    }
    return <Glossary onOpenConcept={setGlossaryModal} />;
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-mark">MM</div>
          {!sidebarCollapsed && (
            <div>
              <div className="brand-name">MentalMetrics</div>
              <div className="brand-subtitle">Clinical EEG review</div>
            </div>
          )}
        </div>

        <nav className="sidebar-nav">
          {APP_PAGES.map((page) => (
            <button
              key={page.key}
              type="button"
              className={`nav-item ${route === page.key ? "active" : ""}`}
              onClick={() => navigate(page.key)}
            >
              <span className="nav-icon">{page.icon}</span>
              {!sidebarCollapsed && <span>{page.label}</span>}
            </button>
          ))}
        </nav>

        <button
          type="button"
          className="ghost-button sidebar-toggle"
          onClick={() => setSidebarCollapsed((current) => !current)}
        >
          {sidebarCollapsed ? "Expand" : "Collapse"}
        </button>
      </aside>

      <div className="app-main-shell">
        <header className="topbar">
          <div>
            <div className="topbar-title">{PAGE_TITLES[route]}</div>
          </div>
          <div className="topbar-actions">
            <button type="button" className="secondary-button" onClick={handleThemeToggle}>
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </button>
            <div className={`status-indicator ${healthStatus}`}>
              <span className="status-dot" />
              <span>{healthStatus === "online" ? "Connected" : healthStatus === "checking" ? "Checking" : "Offline"}</span>
            </div>
          </div>
        </header>

        <main className="content-area">
          {renderPage()}
        </main>
      </div>

      <GlossaryModal conceptName={glossaryModal} onClose={() => setGlossaryModal(null)} />
    </div>
  );
}
