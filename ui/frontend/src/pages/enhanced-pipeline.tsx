import React, { useState } from 'react';
import { Play, Loader2, AlertTriangle, CheckCircle, ChevronDown, ChevronRight, FileText, Brain, Shield, Activity, Zap } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

// Types
interface SectionResult {
    section_id: number;
    detected_type: string;
    classified_type: string;
    classification_score: number;
    content_preview: string;
    domains: Array<{ name: string; score: number }>;
    entities: Array<{ name: string; score: number }>;
    safety: { status: string; finding?: string; score: number };
    relations: Array<{ name: string; score: number }>;
}

interface EnhancedResult {
    note_id: string;
    input_text: string;
    processing_time_ms: number;
    total_sections: number;
    sections: SectionResult[];
    summary: {
        total_sections: number;
        section_types: Array<{ type: string; score: number }>;
        primary_domains: Array<{ name: string; score: number }>;
        safety_alerts: Array<{ section: string; finding: string; status: string }>;
        overall_safety: string;
    };
}

const SAMPLE_NOTE = `* Ajith, 24 yrs ,M, B.Tech software engineer, married, upper middle class, living with wife, informant wife 
* * Information - consistent, adequate and reliable 
* * CC - irritability, anger outburst, sleep disturbances, not doing household works, poor self care * Duration - 6 months * Gradual onset progressive nature * Continuos course 
* * History - married 6 months ago, staying with wife, getting angry over little things, hitting wife, irritable on mother, Sleep disturbance sleep at 2AM, waking up 5AM, not bathing, wearing same clothes 3-4 days, staring at laptop for 4 hours but not productive, 
* * Family history - history of psychiatric illness in mother 
* * Substance use - smoking tobacco, nicotine, alcohol once a month 
* * Premorbid personality - jovial, responsible, respectful towards elders, has many friends * General appearance & behaviour - unkempt, hyper alert, ambulant, not cooperative 
* * PMA - PMA increased, 
* * Talk - QTR increased, RT decreased 
* * Perplexed affect, dysphoric mood 
* * Thought - pre occupied with work and future, worries about job 
* * Perception - normal`;

// Safety status colors
const getSafetyColor = (status: string) => {
    switch (status) {
        case 'alert': return 'bg-red-100 text-red-700 border-red-200';
        case 'monitor': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
        case 'safe': return 'bg-green-100 text-green-700 border-green-200';
        default: return 'bg-gray-100 text-gray-600 border-gray-200';
    }
};

// Section card component
const SectionCard: React.FC<{ section: SectionResult; isExpanded: boolean; onToggle: () => void }> = ({
    section, isExpanded, onToggle
}) => {
    return (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            {/* Header */}
            <button
                onClick={onToggle}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
                <div className="flex items-center space-x-3">
                    {isExpanded ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
                    <FileText className="w-5 h-5 text-blue-500" />
                    <div className="text-left">
                        <span className="font-medium text-gray-900">{section.classified_type}</span>
                        {section.detected_type !== section.classified_type && section.detected_type !== 'Unknown' && (
                            <span className="ml-2 text-xs text-gray-500">(detected: {section.detected_type})</span>
                        )}
                    </div>
                </div>
                <div className="flex items-center space-x-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getSafetyColor(section.safety.status)}`}>
                        {section.safety.status === 'safe' ? '✓ Safe' : section.safety.status === 'alert' ? '⚠️ Alert' : section.safety.status}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${section.classification_score > 0.6 ? 'bg-green-100 text-green-700' :
                            section.classification_score > 0.4 ? 'bg-yellow-100 text-yellow-700' :
                                'bg-gray-100 text-gray-600'
                        }`}>
                        {(section.classification_score * 100).toFixed(0)}%
                    </span>
                </div>
            </button>

            {/* Expanded Content */}
            {isExpanded && (
                <div className="px-4 py-3 border-t border-gray-100 bg-gray-50">
                    {/* Content Preview */}
                    <div className="mb-4">
                        <div className="text-xs font-medium text-gray-500 mb-1">Content</div>
                        <p className="text-sm text-gray-700 bg-white p-2 rounded border border-gray-200">
                            {section.content_preview}
                        </p>
                    </div>

                    {/* Classifications Grid */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                        {/* Domains */}
                        <div className="bg-purple-50 rounded-lg p-3">
                            <div className="flex items-center mb-2">
                                <Brain className="w-4 h-4 text-purple-600 mr-1" />
                                <span className="text-xs font-medium text-purple-700">Domains</span>
                            </div>
                            {section.domains.length > 0 ? (
                                <div className="space-y-1">
                                    {section.domains.map((d, i) => (
                                        <div key={i} className="text-xs flex justify-between">
                                            <span className="text-gray-700 truncate">{d.name}</span>
                                            <span className="text-purple-600 font-medium">{(d.score * 100).toFixed(0)}%</span>
                                        </div>
                                    ))}
                                </div>
                            ) : <span className="text-xs text-gray-500">None</span>}
                        </div>

                        {/* Entities */}
                        <div className="bg-green-50 rounded-lg p-3">
                            <div className="flex items-center mb-2">
                                <Activity className="w-4 h-4 text-green-600 mr-1" />
                                <span className="text-xs font-medium text-green-700">Entities</span>
                            </div>
                            {section.entities.length > 0 ? (
                                <div className="space-y-1">
                                    {section.entities.map((e, i) => (
                                        <div key={i} className="text-xs flex justify-between">
                                            <span className="text-gray-700 truncate">{e.name}</span>
                                            <span className="text-green-600 font-medium">{(e.score * 100).toFixed(0)}%</span>
                                        </div>
                                    ))}
                                </div>
                            ) : <span className="text-xs text-gray-500">None</span>}
                        </div>

                        {/* Safety */}
                        <div className="bg-red-50 rounded-lg p-3">
                            <div className="flex items-center mb-2">
                                <Shield className="w-4 h-4 text-red-600 mr-1" />
                                <span className="text-xs font-medium text-red-700">Safety</span>
                            </div>
                            <div className="text-xs">
                                <div className={`inline-block px-1.5 py-0.5 rounded ${getSafetyColor(section.safety.status)}`}>
                                    {section.safety.finding || section.safety.status}
                                </div>
                            </div>
                        </div>

                        {/* Relations */}
                        <div className="bg-orange-50 rounded-lg p-3">
                            <div className="flex items-center mb-2">
                                <Zap className="w-4 h-4 text-orange-600 mr-1" />
                                <span className="text-xs font-medium text-orange-700">Relations</span>
                            </div>
                            {section.relations.length > 0 ? (
                                <div className="space-y-1">
                                    {section.relations.map((r, i) => (
                                        <div key={i} className="text-xs flex justify-between">
                                            <span className="text-gray-700 truncate">{r.name}</span>
                                            <span className="text-orange-600 font-medium">{(r.score * 100).toFixed(0)}%</span>
                                        </div>
                                    ))}
                                </div>
                            ) : <span className="text-xs text-gray-500">None</span>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export const EnhancedPipelinePage: React.FC = () => {
    const [inputText, setInputText] = useState('');
    const [topK, setTopK] = useState(3);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<EnhancedResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());

    const handleProcess = async () => {
        if (!inputText.trim()) {
            setError('Please enter a clinical note');
            return;
        }

        setIsProcessing(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`${API_BASE_URL}/process/note/enhanced`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText, top_k: topK, use_reranking: false })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            setResult(data);
            // Expand first section by default
            if (data.sections?.length > 0) {
                setExpandedSections(new Set([0]));
            }
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Processing failed');
        } finally {
            setIsProcessing(false);
        }
    };

    const toggleSection = (idx: number) => {
        setExpandedSections(prev => {
            const next = new Set(prev);
            if (next.has(idx)) next.delete(idx);
            else next.add(idx);
            return next;
        });
    };

    const expandAll = () => {
        if (result) setExpandedSections(new Set(result.sections.map((_, i) => i)));
    };

    return (
        <div className="p-6 max-w-6xl mx-auto">
            {/* Header */}
            <div className="mb-6">
                <div className="flex items-center space-x-2 mb-2">
                    <span className="text-2xl">🧠</span>
                    <h1 className="text-2xl font-bold text-gray-900">Enhanced Note Processing</h1>
                </div>
                <p className="text-gray-600">
                    Splits clinical note into sections, then classifies each section through all 5 datasets
                </p>
            </div>

            {/* Input */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 mb-6">
                <div className="flex justify-between mb-3">
                    <h2 className="font-semibold text-gray-900">Clinical Note</h2>
                    <button onClick={() => { setInputText(SAMPLE_NOTE); setResult(null); }}
                        className="text-sm text-indigo-600 hover:text-indigo-800">
                        Load Sample
                    </button>
                </div>
                <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Paste psychiatric clinical note..."
                    className="w-full h-40 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 resize-none font-mono text-sm"
                />
                <div className="mt-3 flex justify-between items-center">
                    <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-500">{inputText.length} chars</span>
                        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                            className="border border-gray-300 rounded px-2 py-1 text-sm">
                            <option value={2}>Top 2</option>
                            <option value={3}>Top 3</option>
                            <option value={5}>Top 5</option>
                        </select>
                    </div>
                    <button onClick={handleProcess} disabled={isProcessing || !inputText.trim()}
                        className="flex items-center px-5 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-400 transition-colors">
                        {isProcessing ? (
                            <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</>
                        ) : (
                            <><Play className="w-4 h-4 mr-2" /> Process Note</>
                        )}
                    </button>
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center">
                    <AlertTriangle className="w-5 h-5 text-red-500 mr-3" />
                    <span className="text-red-700">{error}</span>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="space-y-6">
                    {/* Summary */}
                    <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-5 text-white">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-bold">Analysis Summary</h2>
                            <span className="text-sm opacity-80">{result.processing_time_ms.toFixed(0)}ms | {result.total_sections} sections</span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-white/20 rounded-lg p-3">
                                <div className="text-xs opacity-80 mb-1">Sections Detected</div>
                                <div className="flex flex-wrap gap-1">
                                    {result.summary.section_types.slice(0, 4).map((s, i) => (
                                        <span key={i} className="px-2 py-0.5 bg-white/30 rounded text-xs">{s.type}</span>
                                    ))}
                                </div>
                            </div>
                            <div className="bg-white/20 rounded-lg p-3">
                                <div className="text-xs opacity-80 mb-1">Top Domains</div>
                                <div className="flex flex-wrap gap-1">
                                    {result.summary.primary_domains.slice(0, 3).map((d, i) => (
                                        <span key={i} className="px-2 py-0.5 bg-white/30 rounded text-xs">{d.name}</span>
                                    ))}
                                </div>
                            </div>
                            <div className="bg-white/20 rounded-lg p-3">
                                <div className="text-xs opacity-80 mb-1">Safety Status</div>
                                <div className="flex items-center">
                                    {result.summary.overall_safety.includes('✅') ? (
                                        <CheckCircle className="w-4 h-4 mr-2" />
                                    ) : (
                                        <AlertTriangle className="w-4 h-4 mr-2" />
                                    )}
                                    <span className="text-sm">{result.summary.overall_safety}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Section List */}
                    <div>
                        <div className="flex justify-between items-center mb-3">
                            <h2 className="font-semibold text-gray-900">Section-by-Section Analysis</h2>
                            <button onClick={expandAll} className="text-sm text-indigo-600 hover:text-indigo-800">
                                Expand All
                            </button>
                        </div>
                        <div className="space-y-2">
                            {result.sections.map((section, idx) => (
                                <SectionCard
                                    key={idx}
                                    section={section}
                                    isExpanded={expandedSections.has(idx)}
                                    onToggle={() => toggleSection(idx)}
                                />
                            ))}
                        </div>
                    </div>

                    {/* JSON Output */}
                    <details className="bg-gray-50 rounded-xl border border-gray-200">
                        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700">
                            View Raw JSON
                        </summary>
                        <pre className="p-4 text-xs overflow-auto max-h-80 bg-gray-900 text-gray-100 rounded-b-xl">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </details>
                </div>
            )}
        </div>
    );
};

export default EnhancedPipelinePage;
