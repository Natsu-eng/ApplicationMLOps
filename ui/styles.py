"""
🎨 UI Styles Centralisés - DataLab Pro
Design System Moderne et Réutilisable
Version: 1.0
"""

class UIStyles:
    """Gestionnaire de styles CSS centralisé"""
    
    @staticmethod
    def get_main_css() -> str:
        """Retourne le CSS principal de l'application"""
        return """
        <style>
            /* ============================================
               RESET & BASE
               ============================================ */
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            
            /* ============================================
               HEADERS
               ============================================ */
            .main-header {
                font-size: 2.8rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                margin-bottom: 1rem;
                text-align: center;
                animation: fadeInDown 0.6s ease-out;
                letter-spacing: -1px;
            }
            
            .sub-header {
                text-align: center;
                color: #666;
                font-size: 1.15rem;
                margin-bottom: 2.5rem;
                font-weight: 400;
            }
            
            /* ============================================
               CARDS
               ============================================ */
            .workflow-step-card {
                background: white;
                padding: 2.5rem;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                border: 1px solid rgba(102, 126, 234, 0.1);
                margin-bottom: 2rem;
                animation: fadeIn 0.4s ease-out;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .workflow-step-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
            }
            
            .model-card {
                background: white;
                padding: 1.8rem;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                border: 2px solid transparent;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                height: 100%;
                position: relative;
                overflow: hidden;
            }
            
            .model-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .model-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 28px rgba(102, 126, 234, 0.25);
                border-color: #667eea;
            }
            
            .model-card:hover::before {
                opacity: 1;
            }
            
            .model-card.selected {
                border-color: #667eea;
                background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.35);
            }
            
            .model-card.selected::after {
                content: '✓';
                position: absolute;
                top: 12px;
                right: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 16px;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            /* ============================================
               METRIC CARDS
               ============================================ */
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                transform: rotate(45deg);
            }
            
            .metric-card:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.45);
            }
            
            .metric-card h3 {
                margin: 0;
                font-size: 2.5rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .metric-card h4 {
                margin: 0.5rem 0;
                font-size: 0.95rem;
                opacity: 0.95;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            
            .metric-card h2 {
                margin: 0;
                font-size: 2rem;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* ============================================
               STATUS BADGES
               ============================================ */
            .status-badge {
                display: inline-block;
                padding: 0.4rem 1rem;
                border-radius: 25px;
                font-size: 0.85rem;
                font-weight: 600;
                margin: 0.3rem;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .status-badge:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            
            .badge-success { 
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white; 
            }
            
            .badge-warning { 
                background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
                color: #333; 
            }
            
            .badge-danger { 
                background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                color: white; 
            }
            
            .badge-info { 
                background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
                color: white; 
            }
            
            /* ============================================
               PROGRESS STEPS
               ============================================ */
            .progress-step {
                text-align: center;
                padding: 1.5rem;
                border-radius: 12px;
                transition: all 0.3s ease;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .progress-step.active {
                background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
                border: 2px solid #667eea;
                transform: scale(1.05);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
            }
            
            .progress-step.completed {
                background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                border: 2px solid #28a745;
                box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
            }
            
            .progress-step.pending {
                background: #fafafa;
                border: 2px solid #e0e0e0;
                opacity: 0.7;
            }
            
            /* ============================================
               TASK SELECTION CARDS
               ============================================ */
            .task-card {
                background: white;
                padding: 2.5rem;
                border-radius: 20px;
                border: 3px solid transparent;
                text-align: center;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                height: 240px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                position: relative;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            
            .task-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .task-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.15);
            }
            
            .task-card:hover::before {
                opacity: 1;
            }
            
            .task-card.selected {
                border-color: #667eea;
                background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
                box-shadow: 0 12px 32px rgba(102, 126, 234, 0.35);
                transform: translateY(-5px);
            }
            
            .task-card .icon {
                font-size: 4rem;
                margin-bottom: 1.2rem;
                filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
            }
            
            /* ============================================
               ANIMATIONS
               ============================================ */
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes pulse {
                0%, 100% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1.05);
                }
            }
            
            .pulse {
                animation: pulse 2s infinite;
            }
            
            /* ============================================
               BUTTONS
               ============================================ */
            .stButton > button {
                border-radius: 10px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
            }
            
            .stButton > button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            }
            
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #5568d3 0%, #6a3d91 100%);
            }
            
            /* ============================================
               DATAFRAMES & TABLES
               ============================================ */
            .dataframe {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
            
            /* ============================================
               EXPANDERS
               ============================================ */
            .streamlit-expanderHeader {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 10px;
                font-weight: 600;
                padding: 1rem;
                transition: all 0.2s ease;
            }
            
            .streamlit-expanderHeader:hover {
                background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            }
            
            /* ============================================
               INPUTS & CONTROLS
               ============================================ */
            .stSelectbox, .stSlider, .stCheckbox {
                margin-bottom: 1.2rem;
            }
            
            /* ============================================
               ALERTS
               ============================================ */
            .stAlert {
                border-radius: 10px;
                border-left-width: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            
            /* ============================================
               SIDEBAR
               ============================================ */
            .css-1d391kg {
                background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            }
            
            /* ============================================
               HIDE STREAMLIT BRANDING
               ============================================ */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* ============================================
               TOOLTIPS
               ============================================ */
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: help;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 220px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 8px;
                padding: 8px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.3s;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            
            /* ============================================
               SCROLLBAR CUSTOM
               ============================================ */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #5568d3 0%, #6a3d91 100%);
            }
            
            /* ============================================
               RESPONSIVE
               ============================================ */
            @media (max-width: 768px) {
                .main-header {
                    font-size: 2rem;
                }
                
                .workflow-step-card {
                    padding: 1.5rem;
                }
                
                .task-card {
                    height: auto;
                    padding: 2rem;
                }
            }
        </style>
        """
    
    @staticmethod
    def render_progress_bar(progress: float, step: int, total_steps: int) -> str:
        """Génère une barre de progression moderne"""
        return f"""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                Progression
            </div>
            <div style="background: #e0e0e0; border-radius: 10px; height: 10px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            width: {progress}%; height: 100%; transition: width 0.5s ease;
                            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);"></div>
            </div>
            <div style="font-size: 0.85rem; color: #667eea; margin-top: 0.5rem; font-weight: 600;">
                Étape {step}/{total_steps}
            </div>
        </div>
        """
    
    @staticmethod
    def render_system_metrics(memory_percent: float) -> str:
        """Génère un widget de métriques système"""
        memory_color = "#28a745" if memory_percent < 70 else "#ffc107" if memory_percent < 85 else "#dc3545"
        
        return f"""
        <div style="text-align: center;">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                Système
            </div>
            <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem;">
                <div style="width: 50px; height: 50px; border-radius: 50%; 
                            background: {memory_color}; display: flex; align-items: center; 
                            justify-content: center; color: white; font-weight: bold;
                            box-shadow: 0 4px 12px {memory_color}40;">
                    {memory_percent:.0f}
                </div>
                <div style="text-align: left;">
                    <div style="font-size: 0.85rem; color: #666; font-weight: 600;">RAM</div>
                    <div style="font-size: 0.75rem; color: #999;">
                        {100 - memory_percent:.0f}% libre
                    </div>
                </div>
            </div>
        </div>
        """