"""
🎨 Styles CSS Évaluation ML - Design Moderne avec Tableaux Complexes
Version: 3.0
"""

class EvaluationStyles:
    """Styles CSS optimisés pour la page d'évaluation avec tableaux complexes"""
    
    @staticmethod
    def get_evaluation_css() -> str:
        return """
        <style>
            /* ============================================
               MÉTRIQUES HORIZONTALES COMPACTES
               ============================================ */
            .metrics-horizontal-compact {
                display: flex !important;
                flex-wrap: nowrap !important;
                overflow-x: auto;
                gap: 1rem;
                padding: 1rem 0;
                scrollbar-width: thin;
            }
            
            .metric-card-horizontal {
                background: white;
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card-horizontal::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: var(--card-color, #667eea);
            }
            
            .metric-card-horizontal:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            }
            
            .metric-icon-horizontal {
                font-size: 1.8rem;
                margin-bottom: 0.3rem;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            }
            
            .metric-value-horizontal {
                font-size: 1.5rem;
                font-weight: 800;
                color: #2d3748;
                margin: 0.2rem 0;
            }
            
            .metric-label-horizontal {
                font-size: 0.75rem;
                color: #6c757d;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .metric-subtitle-horizontal {
                font-size: 0.7rem;
                color: #a0aec0;
                margin-top: 0.2rem;
            }
            
            /* ============================================
               TABLEAUX COMPLEXES STYLÉS
               ============================================ */
            .complex-table-container {
                background: white;
                border-radius: 15px;
                padding: 0;
                margin: 1.5rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                overflow: hidden;
                border: 1px solid #e9ecef;
            }
            
            .table-header-modern {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.2rem 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .table-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin: 0;
            }
            
            .table-subtitle {
                font-size: 0.85rem;
                opacity: 0.9;
            }
            
            .complex-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
            }
            
            .complex-table thead tr {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-bottom: 2px solid #dee2e6;
            }
            
            .complex-table th {
                padding: 1rem;
                text-align: left;
                font-weight: 700;
                color: #495057;
                text-transform: uppercase;
                font-size: 0.8rem;
                letter-spacing: 0.5px;
                border-right: 1px solid #dee2e6;
            }
            
            .complex-table th:last-child {
                border-right: none;
            }
            
            .complex-table td {
                padding: 0.8rem 1rem;
                border-bottom: 1px solid #e9ecef;
                border-right: 1px solid #f8f9fa;
            }
            
            .complex-table td:last-child {
                border-right: none;
            }
            
            .complex-table tbody tr {
                transition: all 0.3s ease;
            }
            
            .complex-table tbody tr:hover {
                background: rgba(102, 126, 234, 0.05);
                transform: scale(1.002);
            }
            
            .complex-table tbody tr.best-model-row {
                background: linear-gradient(90deg, rgba(40, 167, 69, 0.08) 0%, transparent 100%);
                border-left: 4px solid #28a745;
            }
            
            .complex-table tbody tr.failed-model-row {
                background: linear-gradient(90deg, rgba(220, 53, 69, 0.06) 0%, transparent 100%);
                border-left: 4px solid #dc3545;
            }
            
            /* ============================================
               BADGES ET INDICATEURS AVANCÉS
               ============================================ */
            .metric-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.3rem 0.8rem;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 600;
                gap: 0.3rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
            }
            
            .metric-badge:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            }
            
            .badge-excellent {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
            }
            
            .badge-good {
                background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
                color: white;
            }
            
            .badge-fair {
                background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
                color: #333;
            }
            
            .badge-poor {
                background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                color: white;
            }
            
            .badge-neutral {
                background: #6c757d;
                color: white;
            }
            
            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                padding: 0.3rem 0.8rem;
                border-radius: 8px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            
            .status-success {
                background: rgba(40, 167, 69, 0.15);
                color: #28a745;
                border: 1px solid rgba(40, 167, 69, 0.3);
            }
            
            .status-failed {
                background: rgba(220, 53, 69, 0.15);
                color: #dc3545;
                border: 1px solid rgba(220, 53, 69, 0.3);
            }
            
            .status-warning {
                background: rgba(255, 193, 7, 0.15);
                color: #856404;
                border: 1px solid rgba(255, 193, 7, 0.3);
            }
            
            /* ============================================
               PROGRESS BARS ET INDICATEURS VISUELS
               ============================================ */
            .progress-container {
                background: #e9ecef;
                border-radius: 10px;
                height: 8px;
                overflow: hidden;
                margin: 0.3rem 0;
            }
            
            .progress-bar {
                height: 100%;
                border-radius: 10px;
                transition: width 0.5s ease;
            }
            
            .progress-excellent { background: linear-gradient(90deg, #28a745, #20c997); }
            .progress-good { background: linear-gradient(90deg, #17a2b8, #138496); }
            .progress-fair { background: linear-gradient(90deg, #ffc107, #ff9800); }
            .progress-poor { background: linear-gradient(90deg, #dc3545, #c82333); }
            
            /* ============================================
               SECTIONS ET CONTAINERS
               ============================================ */
            .section-header {
                font-size: 1.3rem;
                font-weight: 700;
                color: #2d3748;
                margin: 1.5rem 0 1rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e9ecef;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .section-header::before {
                content: '';
                width: 4px;
                height: 20px;
                background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
                border-radius: 2px;
            }
            
            .plot-container-modern {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
                margin: 1rem 0;
            }
            
            /* ============================================
               ONGLETS MODERNES
               ============================================ */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 0.5rem;
                border-radius: 15px;
                margin-bottom: 2rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre;
                background: transparent;
                border-radius: 10px;
                gap: 0.5rem;
                padding: 0 1.5rem;
                font-weight: 600;
                color: #6c757d;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: white !important;
                color: #667eea !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            /* ============================================
               RESPONSIVE
               ============================================ */
            @media (max-width: 768px) {
                .metrics-horizontal-compact {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .complex-table-container {
                    overflow-x: auto;
                }
                
                .complex-table {
                    min-width: 800px;
                }
            }
        </style>
        """