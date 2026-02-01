import React, { useState } from 'react';
import { Play, Loader2, AlertTriangle, CheckCircle, Brain, FileText, Shield, Activity, Link } from 'lucide-react';

// API Base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

// Result types
interface ClassificationItem {
    name: string;
    description: string;
    score: number;
    reasoning?: string;
}

interface CategoryResult {
    category: string;
    dataset_id: string;
    dataset_name: string;
    top_result: ClassificationItem;
    alternatives: ClassificationItem[];
    processing_time_ms: number;
}

interface PipelineResult {
    note_id: string;
    input_text: string;
    processing_time_ms: number;
    results: Record<string, CategoryResult>;
    summary: {
        primary_section: string | null;
        primary_domain: string | null;
        safety_status: string | null;
        key_findings: Array<{ category: string; finding: string; confidence: string }>;
    };
}

// Category icons and colors
const CATEGORY_CONFIG: Record<string, { icon: React.ElementType; color: string; bgColor: string }> = {
    sections: { icon: FileText, color: 'text-blue-600', bgColor: 'bg-blue-100' },
    domains: { icon: Brain, color: 'text-purple-600', bgColor: 'bg-purple-100' },
    entities: { icon: Activity, color: 'text-green-600', bgColor: 'bg-green-100' },
    safety: { icon: Shield, color: 'text-red-600', bgColor: 'bg-red-100' },
    relations: { icon: Link, color: 'text-orange-600', bgColor: 'bg-orange-100' },
};

// Sample clinical note
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

export const PipelinePage: React.FC = () => {
    const [inputText, setInputText] = useState('');
    const [topK, setTopK] = useState(3);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<PipelineResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleProcess = async () => {
        if (!inputText.trim()) {
            setError('Please enter a clinical note');
            return;
        }

        setIsProcessing(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`${API_BASE_URL}/process/note`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: inputText,
                    top_k: topK,
                    use_reranking: false
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Processing failed');
        } finally {
            setIsProcessing(false);
        }
    };

    const loadSample = () => {
        setInputText(SAMPLE_NOTE);
        setResult(null);
        setError(null);
    };

    return (
        <div className="p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Process Clinical Note</h1>
                <p className="mt-2 text-gray-600">
                    Run all 5 classifiers on a psychiatric clinical note and get structured output
                </p>
            </div>

            {/* Input Section */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-900">Clinical Note Input</h2>
                    <button
                        onClick={loadSample}
                        className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                    >
                        Load Sample Note
                    </button>
                </div>

                <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Paste or type a psychiatric clinical note here..."
                    className="w-full h-48 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none font-mono text-sm"
                />

                <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <label className="text-sm text-gray-600">Top K Results:</label>
                        <select
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            className="border border-gray-300 rounded-md px-3 py-1.5 text-sm"
                        >
                            <option value={3}>3</option>
                            <option value={5}>5</option>
                            <option value={10}>10</option>
                        </select>
                    </div>

                    <button
                        onClick={handleProcess}
                        disabled={isProcessing || !inputText.trim()}
                        className="flex items-center px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                    >
                        {isProcessing ? (
                            <>
                                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            <>
                                <Play className="w-5 h-5 mr-2" />
                                Process Note
                            </>
                        )}
                    </button>
                </div>

                <div className="mt-2 text-sm text-gray-500">
                    {inputText.length} characters
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center">
                    <AlertTriangle className="w-5 h-5 text-red-500 mr-3" />
                    <span className="text-red-700">{error}</span>
                </div>
            )}

            {/* Results Section */}
            {result && (
                <div className="space-y-6">
                    {/* Summary Card */}
                    <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-xl font-bold">Classification Summary</h2>
                            <span className="text-sm opacity-80">
                                Processed in {result.processing_time_ms.toFixed(0)}ms
                            </span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-white/20 rounded-lg p-4">
                                <div className="text-sm opacity-80">Primary Section</div>
                                <div className="text-lg font-semibold">{result.summary.primary_section || 'N/A'}</div>
                            </div>
                            <div className="bg-white/20 rounded-lg p-4">
                                <div className="text-sm opacity-80">Primary Domain</div>
                                <div className="text-lg font-semibold">{result.summary.primary_domain || 'N/A'}</div>
                            </div>
                            <div className="bg-white/20 rounded-lg p-4">
                                <div className="text-sm opacity-80">Safety Status</div>
                                <div className="text-lg font-semibold flex items-center">
                                    {result.summary.safety_status?.includes('⚠️') ? (
                                        <AlertTriangle className="w-5 h-5 mr-2" />
                                    ) : (
                                        <CheckCircle className="w-5 h-5 mr-2" />
                                    )}
                                    {result.summary.safety_status || 'N/A'}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Category Results */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {Object.entries(result.results).map(([category, catResult]) => {
                            const config = CATEGORY_CONFIG[category] || CATEGORY_CONFIG.sections;
                            const Icon = config.icon;

                            return (
                                <div key={category} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                                    {/* Category Header */}
                                    <div className={`${config.bgColor} px-4 py-3 flex items-center justify-between`}>
                                        <div className="flex items-center">
                                            <Icon className={`w-5 h-5 ${config.color} mr-2`} />
                                            <span className={`font-semibold ${config.color}`}>{catResult.dataset_name}</span>
                                        </div>
                                        <span className="text-sm text-gray-500">{catResult.processing_time_ms.toFixed(0)}ms</span>
                                    </div>

                                    {/* Top Result */}
                                    <div className="p-4 border-b border-gray-100">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-semibold text-gray-900">{catResult.top_result.name}</span>
                                            <span className={`px-2 py-1 rounded text-sm font-medium ${catResult.top_result.score > 0.5 ? 'bg-green-100 text-green-700' :
                                                    catResult.top_result.score > 0.3 ? 'bg-yellow-100 text-yellow-700' :
                                                        'bg-gray-100 text-gray-600'
                                                }`}>
                                                {(catResult.top_result.score * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-600">{catResult.top_result.description}</p>
                                    </div>

                                    {/* Alternatives */}
                                    {catResult.alternatives.length > 0 && (
                                        <div className="px-4 py-3 bg-gray-50">
                                            <div className="text-xs text-gray-500 mb-2">Other candidates:</div>
                                            <div className="space-y-1">
                                                {catResult.alternatives.slice(0, 2).map((alt, idx) => (
                                                    <div key={idx} className="flex items-center justify-between text-sm">
                                                        <span className="text-gray-700">{alt.name}</span>
                                                        <span className="text-gray-500">{(alt.score * 100).toFixed(0)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Raw JSON Output */}
                    <details className="bg-gray-50 rounded-xl border border-gray-200">
                        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                            View Raw JSON Output
                        </summary>
                        <pre className="p-4 text-xs overflow-auto max-h-96 bg-gray-900 text-gray-100 rounded-b-xl">
                            {JSON.stringify(result, null, 2)}
                        </pre>
                    </details>
                </div>
            )}
        </div>
    );
};

export default PipelinePage;
