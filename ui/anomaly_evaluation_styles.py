"""
🎨 Styles CSS pour la page d'évaluation d'anomalies
Extrait de 5_anomaly_evaluation.py pour centralisation
"""

class AnomalyEvaluationStyles:
    """Styles CSS pour le dashboard d'évaluation premium"""
    
    @staticmethod
    def get_css() -> str:
        """Retourne le CSS complet pour la page d'évaluation"""
        return """
<style>
    /* Variables */
    :root {
        --primary: #6366f1;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --bg-card: #ffffff;
        --shadow: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Base */
    .block-container {
        padding: 1.5rem 2.5rem !important;
        max-width: 1600px;
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards Premium */
    .metric-card-premium {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid #e5e7eb;
        transition: box-shadow 0.2s ease;
        position: relative;
    }
    .metric-card-premium:hover {
        box-shadow: var(--shadow-lg);
    }
    .metric-card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: var(--primary);
        border-radius: 12px 12px 0 0;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-trend {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .trend-up {
        background: #d1fae5;
        color: #065f46;
    }
    
    .trend-down {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-excellent {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    .badge-good {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border: 2px solid #3b82f6;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 2px solid #f59e0b;
    }
    
    .badge-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    /* Panel Cards */
    .panel-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    .panel-icon {
        font-size: 1.75rem;
    }
    
    .panel-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow);
    }
    
    .rec-priority-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #ef4444;
    }
    
    .rec-priority-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }
    
    .rec-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Error Analysis */
    .error-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .error-fp {
        background: #fef2f2;
        border-color: #fca5a5;
    }
    
    .error-fn {
        background: #fef3c7;
        border-color: #fcd34d;
    }
    
    .error-tp {
        background: #d1fae5;
        border-color: #6ee7b7;
    }
    
    /* Progress Indicator */
    .progress-wrapper {
        background: #f3f4f6;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }
    
    .progress-excellent {
        background: linear-gradient(90deg, #10b981, #059669);
    }
    
    .progress-good {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
    }
    
    .progress-warning {
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
    
    .progress-critical {
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    /* Tabs Moderne */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f9fafb;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: var(--shadow);
    }
    
    /* Images Gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-item {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
        transition: transform 0.3s ease;
    }
    
    .image-item:hover {
        transform: scale(1.05);
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 12px;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1f2937;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        .hero-title {
            font-size: 1.75rem;
        }
        .metric-value {
            font-size: 1.75rem;
        }
    }
</style>
"""


