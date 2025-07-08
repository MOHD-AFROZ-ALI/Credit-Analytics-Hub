"""
Compliance Utilities Module for Credit Analytics Hub

This module provides regulatory compliance utilities including
audit trails, compliance checking, regulatory reporting, and
documentation for credit risk management systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Enumeration for compliance levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    CRITICAL = "critical"

class RegulationType(Enum):
    """Enumeration for regulation types."""
    BASEL_III = "basel_iii"
    CECL = "cecl"
    GDPR = "gdpr"
    FAIR_LENDING = "fair_lending"
    FCRA = "fcra"
    ECOA = "ecoa"

@dataclass
class ComplianceCheck:
    """Data class for individual compliance checks."""
    check_id: str
    regulation: RegulationType
    description: str
    status: ComplianceLevel
    details: Dict[str, Any]
    timestamp: datetime
    remediation_required: bool = False
    remediation_steps: List[str] = None

@dataclass
class ComplianceReport:
    """Data class for compliance reports."""
    report_id: str
    report_type: str
    compliance_level: ComplianceLevel
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    checks: List[ComplianceCheck]
    generated_at: datetime
    generated_by: str
    summary: Dict[str, Any]

class AuditTrail:
    """
    Audit trail utility for tracking model and data operations.
    
    Provides comprehensive logging and tracking of all operations
    for regulatory compliance and audit purposes.
    """
    
    def __init__(self, audit_file_path: str = "/home/user/output/audit_trail.json"):
        """
        Initialize audit trail.
        
        Args:
            audit_file_path: Path to audit trail file
        """
        self.audit_file_path = Path(audit_file_path)
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_entries = []
        self._load_existing_audit_trail()
    
    def _load_existing_audit_trail(self):
        """Load existing audit trail from file."""
        if self.audit_file_path.exists():
            try:
                with open(self.audit_file_path, 'r') as f:
                    data = json.load(f)
                    self.audit_entries = data.get('audit_entries', [])
                logger.info(f"Loaded {len(self.audit_entries)} existing audit entries")
            except Exception as e:
                logger.error(f"Error loading audit trail: {e}")
                self.audit_entries = []
    
    def log_operation(self, 
                     operation_type: str,
                     operation_details: Dict[str, Any],
                     user_id: str = "system",
                     data_hash: Optional[str] = None,
                     model_version: Optional[str] = None):
        """
        Log an operation to the audit trail.
        
        Args:
            operation_type: Type of operation (e.g., 'data_load', 'model_train', 'prediction')
            operation_details: Details about the operation
            user_id: ID of user performing operation
            data_hash: Hash of data involved in operation
            model_version: Version of model used
        """
        entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'operation_details': operation_details,
            'user_id': user_id,
            'data_hash': data_hash,
            'model_version': model_version,
            'ip_address': 'localhost',  # In real implementation, get actual IP
            'session_id': str(uuid.uuid4())[:8]
        }
        
        self.audit_entries.append(entry)
        self._save_audit_trail()
        
        logger.info(f"Audit entry logged: {operation_type} by {user_id}")
    
    def _save_audit_trail(self):
        """Save audit trail to file."""
        audit_data = {
            'audit_entries': self.audit_entries,
            'last_updated': datetime.now().isoformat(),
            'total_entries': len(self.audit_entries)
        }
        
        try:
            with open(self.audit_file_path, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving audit trail: {e}")
    
    def get_audit_summary(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         operation_type: Optional[str] = None,
                         user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get audit trail summary with optional filtering.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            operation_type: Filter by operation type
            user_id: Filter by user ID
            
        Returns:
            Audit summary dictionary
        """
        filtered_entries = self.audit_entries.copy()
        
        # Apply filters
        if start_date:
            filtered_entries = [e for e in filtered_entries 
                              if datetime.fromisoformat(e['timestamp']) >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries 
                              if datetime.fromisoformat(e['timestamp']) <= end_date]
        
        if operation_type:
            filtered_entries = [e for e in filtered_entries 
                              if e['operation_type'] == operation_type]
        
        if user_id:
            filtered_entries = [e for e in filtered_entries 
                              if e['user_id'] == user_id]
        
        # Calculate summary statistics
        operation_counts = {}
        user_counts = {}
        
        for entry in filtered_entries:
            op_type = entry['operation_type']
            user = entry['user_id']
            
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
            user_counts[user] = user_counts.get(user, 0) + 1
        
        return {
            'total_entries': len(filtered_entries),
            'date_range': {
                'start': min([e['timestamp'] for e in filtered_entries]) if filtered_entries else None,
                'end': max([e['timestamp'] for e in filtered_entries]) if filtered_entries else None
            },
            'operation_counts': operation_counts,
            'user_counts': user_counts,
            'most_active_user': max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else None,
            'most_common_operation': max(operation_counts.items(), key=lambda x: x[1])[0] if operation_counts else None
        }
    
    def export_audit_report(self, 
                           output_path: str,
                           format_type: str = 'json',
                           **filter_kwargs) -> str:
        """
        Export audit report in specified format.
        
        Args:
            output_path: Path for output file
            format_type: Format type ('json', 'csv', 'excel')
            **filter_kwargs: Filtering arguments
            
        Returns:
            Path to exported file
        """
        summary = self.get_audit_summary(**filter_kwargs)
        
        if format_type == 'json':
            export_data = {
                'audit_summary': summary,
                'audit_entries': self.audit_entries,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format_type == 'csv':
            df = pd.DataFrame(self.audit_entries)
            df.to_csv(output_path, index=False)
        
        elif format_type == 'excel':
            df = pd.DataFrame(self.audit_entries)
            with pd.ExcelWriter(output_path) as writer:
                df.to_excel(writer, sheet_name='Audit_Entries', index=False)
                pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Audit report exported to {output_path}")
        return output_path

class ComplianceChecker:
    """
    Comprehensive compliance checking utility.
    
    Performs various compliance checks for credit risk models
    and data processing according to regulatory requirements.
    """
    
    def __init__(self, regulations: List[RegulationType] = None):
        """
        Initialize compliance checker.
        
        Args:
            regulations: List of regulations to check against
        """
        self.regulations = regulations or [RegulationType.FAIR_LENDING, RegulationType.GDPR]
        self.compliance_rules = self._initialize_compliance_rules()
        self.audit_trail = AuditTrail()
    
    def _initialize_compliance_rules(self) -> Dict[RegulationType, Dict[str, Any]]:
        """Initialize compliance rules for different regulations."""
        rules = {
            RegulationType.FAIR_LENDING: {
                'protected_attributes': ['race', 'gender', 'age', 'religion', 'national_origin'],
                'disparate_impact_threshold': 0.8,  # 80% rule
                'documentation_required': True,
                'model_validation_required': True
            },
            RegulationType.GDPR: {
                'data_retention_days': 2555,  # 7 years
                'consent_required': True,
                'right_to_explanation': True,
                'data_minimization': True,
                'anonymization_required': False
            },
            RegulationType.FCRA: {
                'adverse_action_threshold': 0.5,
                'notification_required': True,
                'score_disclosure_required': True,
                'dispute_process_required': True
            },
            RegulationType.BASEL_III: {
                'capital_adequacy_ratio': 0.08,
                'stress_testing_required': True,
                'model_validation_frequency_days': 365,
                'backtesting_required': True
            }
        }
        return rules
    
    def check_data_compliance(self, 
                            data: pd.DataFrame,
                            metadata: Dict[str, Any] = None) -> List[ComplianceCheck]:
        """
        Check data compliance against regulations.
        
        Args:
            data: Dataset to check
            metadata: Additional metadata about the data
            
        Returns:
            List of compliance check results
        """
        checks = []
        metadata = metadata or {}
        
        for regulation in self.regulations:
            if regulation == RegulationType.FAIR_LENDING:
                checks.extend(self._check_fair_lending_data(data, metadata))
            elif regulation == RegulationType.GDPR:
                checks.extend(self._check_gdpr_data(data, metadata))
            elif regulation == RegulationType.FCRA:
                checks.extend(self._check_fcra_data(data, metadata))
        
        # Log compliance check
        self.audit_trail.log_operation(
            operation_type='compliance_check',
            operation_details={
                'check_type': 'data_compliance',
                'regulations_checked': [r.value for r in self.regulations],
                'data_shape': data.shape,
                'total_checks': len(checks),
                'passed_checks': len([c for c in checks if c.status == ComplianceLevel.COMPLIANT])
            }
        )
        
        return checks
    
    def _check_fair_lending_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[ComplianceCheck]:
        """Check fair lending compliance for data."""
        checks = []
        rules = self.compliance_rules[RegulationType.FAIR_LENDING]
        
        # Check for protected attributes
        protected_attrs_found = []
        for attr in rules['protected_attributes']:
            if attr in data.columns:
                protected_attrs_found.append(attr)
        
        if protected_attrs_found:
            checks.append(ComplianceCheck(
                check_id=f"FL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.FAIR_LENDING,
                description="Protected attributes found in dataset",
                status=ComplianceLevel.WARNING,
                details={
                    'protected_attributes_found': protected_attrs_found,
                    'recommendation': 'Ensure protected attributes are not used in model training'
                },
                timestamp=datetime.now(),
                remediation_required=True,
                remediation_steps=[
                    'Remove protected attributes from model features',
                    'Implement bias testing procedures',
                    'Document fair lending compliance measures'
                ]
            ))
        else:
            checks.append(ComplianceCheck(
                check_id=f"FL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.FAIR_LENDING,
                description="No protected attributes found in dataset",
                status=ComplianceLevel.COMPLIANT,
                details={'protected_attributes_found': []},
                timestamp=datetime.now()
            ))
        
        return checks
    
    def _check_gdpr_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[ComplianceCheck]:
        """Check GDPR compliance for data."""
        checks = []
        rules = self.compliance_rules[RegulationType.GDPR]
        
        # Check for PII columns
        pii_indicators = ['name', 'email', 'phone', 'address', 'ssn', 'id', 'passport']
        pii_columns = []
        
        for col in data.columns:
            if any(indicator in col.lower() for indicator in pii_indicators):
                pii_columns.append(col)
        
        if pii_columns:
            checks.append(ComplianceCheck(
                check_id=f"GDPR_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.GDPR,
                description="Potential PII columns detected",
                status=ComplianceLevel.WARNING,
                details={
                    'pii_columns': pii_columns,
                    'recommendation': 'Ensure proper consent and data protection measures'
                },
                timestamp=datetime.now(),
                remediation_required=True,
                remediation_steps=[
                    'Verify consent for data processing',
                    'Implement data anonymization if possible',
                    'Ensure data retention policies are followed',
                    'Provide mechanism for data subject rights'
                ]
            ))
        else:
            checks.append(ComplianceCheck(
                check_id=f"GDPR_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.GDPR,
                description="No obvious PII columns detected",
                status=ComplianceLevel.COMPLIANT,
                details={'pii_columns': []},
                timestamp=datetime.now()
            ))
        
        return checks
    
    def _check_fcra_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> List[ComplianceCheck]:
        """Check FCRA compliance for data."""
        checks = []
        
        # Check if credit score is present (FCRA applies to credit decisions)
        credit_score_cols = [col for col in data.columns if 'credit' in col.lower() and 'score' in col.lower()]
        
        if credit_score_cols:
            checks.append(ComplianceCheck(
                check_id=f"FCRA_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.FCRA,
                description="Credit scoring data detected - FCRA compliance required",
                status=ComplianceLevel.WARNING,
                details={
                    'credit_score_columns': credit_score_cols,
                    'recommendation': 'Ensure FCRA compliance procedures are in place'
                },
                timestamp=datetime.now(),
                remediation_required=True,
                remediation_steps=[
                    'Implement adverse action notice procedures',
                    'Provide credit score disclosure mechanisms',
                    'Establish dispute resolution process',
                    'Ensure accuracy and completeness of credit data'
                ]
            ))
        
        return checks
    
    def check_model_compliance(self, 
                             model_metadata: Dict[str, Any],
                             performance_metrics: Dict[str, float] = None) -> List[ComplianceCheck]:
        """
        Check model compliance against regulations.
        
        Args:
            model_metadata: Metadata about the model
            performance_metrics: Model performance metrics
            
        Returns:
            List of compliance check results
        """
        checks = []
        performance_metrics = performance_metrics or {}
        
        for regulation in self.regulations:
            if regulation == RegulationType.FAIR_LENDING:
                checks.extend(self._check_fair_lending_model(model_metadata, performance_metrics))
            elif regulation == RegulationType.BASEL_III:
                checks.extend(self._check_basel_model(model_metadata, performance_metrics))
        
        return checks
    
    def _check_fair_lending_model(self, model_metadata: Dict[str, Any], performance_metrics: Dict[str, float]) -> List[ComplianceCheck]:
        """Check fair lending compliance for model."""
        checks = []
        
        # Check if bias testing has been performed
        bias_testing = model_metadata.get('bias_testing_performed', False)
        
        if not bias_testing:
            checks.append(ComplianceCheck(
                check_id=f"FL_MODEL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.FAIR_LENDING,
                description="Bias testing not documented",
                status=ComplianceLevel.NON_COMPLIANT,
                details={'bias_testing_performed': bias_testing},
                timestamp=datetime.now(),
                remediation_required=True,
                remediation_steps=[
                    'Perform disparate impact analysis',
                    'Test model performance across protected groups',
                    'Document bias testing procedures and results'
                ]
            ))
        else:
            checks.append(ComplianceCheck(
                check_id=f"FL_MODEL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                regulation=RegulationType.FAIR_LENDING,
                description="Bias testing documented",
                status=ComplianceLevel.COMPLIANT,
                details={'bias_testing_performed': bias_testing},
                timestamp=datetime.now()
            ))
        
        return checks
    
    def _check_basel_model(self, model_metadata: Dict[str, Any], performance_metrics: Dict[str, float]) -> List[ComplianceCheck]:
        """Check Basel III compliance for model."""
        checks = []
        
        # Check model validation frequency
        last_validation = model_metadata.get('last_validation_date')
        if last_validation:
            last_validation_date = datetime.fromisoformat(last_validation) if isinstance(last_validation, str) else last_validation
            days_since_validation = (datetime.now() - last_validation_date).days
            
            if days_since_validation > 365:
                checks.append(ComplianceCheck(
                    check_id=f"BASEL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    regulation=RegulationType.BASEL_III,
                    description="Model validation overdue",
                    status=ComplianceLevel.NON_COMPLIANT,
                    details={
                        'days_since_validation': days_since_validation,
                        'last_validation_date': last_validation
                    },
                    timestamp=datetime.now(),
                    remediation_required=True,
                    remediation_steps=[
                        'Perform comprehensive model validation',
                        'Update model documentation',
                        'Review model performance and stability'
                    ]
                ))
            else:
                checks.append(ComplianceCheck(
                    check_id=f"BASEL_001_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    regulation=RegulationType.BASEL_III,
                    description="Model validation up to date",
                    status=ComplianceLevel.COMPLIANT,
                    details={
                        'days_since_validation': days_since_validation,
                        'last_validation_date': last_validation
                    },
                    timestamp=datetime.now()
                ))
        
        return checks

class RegulatoryReporter:
    """
    Regulatory reporting utility.
    
    Generates comprehensive compliance reports for various
    regulatory requirements and audit purposes.
    """
    
    def __init__(self, output_dir: str = "/home/user/output/compliance_reports"):
        """
        Initialize regulatory reporter.
        
        Args:
            output_dir: Directory for compliance reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compliance_checker = ComplianceChecker()
    
    def generate_compliance_report(self, 
                                 data: pd.DataFrame,
                                 model_metadata: Dict[str, Any] = None,
                                 report_type: str = "comprehensive") -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            data: Dataset to analyze
            model_metadata: Model metadata for compliance checking
            report_type: Type of report to generate
            
        Returns:
            Compliance report object
        """
        report_id = f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Perform compliance checks
        data_checks = self.compliance_checker.check_data_compliance(data)
        model_checks = []
        
        if model_metadata:
            model_checks = self.compliance_checker.check_model_compliance(model_metadata)
        
        all_checks = data_checks + model_checks
        
        # Calculate summary statistics
        total_checks = len(all_checks)
        passed_checks = len([c for c in all_checks if c.status == ComplianceLevel.COMPLIANT])
        failed_checks = len([c for c in all_checks if c.status == ComplianceLevel.NON_COMPLIANT])
        warning_checks = len([c for c in all_checks if c.status == ComplianceLevel.WARNING])
        
        # Determine overall compliance level
        if failed_checks > 0:
            overall_compliance = ComplianceLevel.NON_COMPLIANT
        elif warning_checks > 0:
            overall_compliance = ComplianceLevel.WARNING
        else:
            overall_compliance = ComplianceLevel.COMPLIANT
        
        # Create summary
        summary = {
            'data_shape': data.shape,
            'regulations_checked': [r.value for r in self.compliance_checker.regulations],
            'compliance_score': passed_checks / total_checks if total_checks > 0 else 1.0,
            'critical_issues': failed_checks,
            'warnings': warning_checks,
            'recommendations_count': len([c for c in all_checks if c.remediation_required])
        }
        
        report = ComplianceReport(
            report_id=report_id,
            report_type=report_type,
            compliance_level=overall_compliance,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            checks=all_checks,
            generated_at=datetime.now(),
            generated_by="system",
            summary=summary
        )
        
        return report
    
    def save_compliance_report(self, 
                             report: ComplianceReport,
                             format_type: str = "json") -> str:
        """
        Save compliance report to file.
        
        Args:
            report: Compliance report to save
            format_type: Format for saving ('json', 'html', 'pdf')
            
        Returns:
            Path to saved report
        """
        timestamp = report.generated_at.strftime('%Y%m%d_%H%M%S')
        
        if format_type == "json":
            filename = f"compliance_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert report to dictionary
            report_dict = asdict(report)
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=convert_datetime)
        
        elif format_type == "html":
            filename = f"compliance_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            html_content = self._generate_html_report(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info(f"Compliance report saved to {filepath}")
        return str(filepath)
    
    # def _generate_html_report(self, report: ComplianceReport) -> str:
    #     """Generate HTML compliance report."""
    #     html = f"""
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <title>Compliance Report - {report.report_id}</title>
    #         <style>
    #             body {{ font-family: Arial, sans-serif; margin: 20px; }}
    #             .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
    #             .summary {{ background-color: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    #             .compliant {{ color: #28a745; }}
    #             .warning {{ color: #ffc107; }}
    #             .non-compliant {{ color: #dc3545; }}
    #             .check {{ margin: 10px 0; padding: 10px; border-left: 4px solid #dee2e6; }}
    #             .check.compliant {{ border-left-color: #28a745; }}
    #             .check.warning {{ border-left-color: #ffc107; }}
    #             .check.non-compliant {{ border-left-color: #dc3545; }}
    #         </style>
    #     </head>
    #     <body>
    #         <div class="header">
    #             <h1>Compliance Report</h1>
    #             <p><strong>Report ID:</strong> {report.report_id}</p>
    #             <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    #             <p><strong>Overall Status:</strong> <span class="{report.compliance_level.value}">{report.compliance_level.value.upper()}</span></p>
    #         </div>
            
    #         <div class="summary">
    #             <h2>Summary</h2>
    #             <p><strong>Total Checks:</strong> {report.total_checks}</p>
    #             <p><strong>Passed:</strong> <span class="compliant">{report.passed_checks}</span></p>
    #             <p><strong>Warnings:</strong> <span class="warning">{report.warning_checks}</span></p>
    #             <p><strong>Failed:</strong> <span class="non-compliant">{report.failed_checks}</span></p>
    #             <p><strong>Compliance Score:</strong> {report.summary['compliance_score']:.2%}</p>
    #         </div>
            
    #         <h2>Detailed Checks</h2>
    #     """
        
    #     for check in report.checks:
    #         status_class = check.status.value.replace('_', '-')
    #         html += f"""
    #         <div class="check {status_class}">
    #             <h3>{check.description}</h3>
    #             <p><strong>Regulation:</strong> {check.regulation.value.upper()}</p>
    #             <p><strong>Status:</strong> <span class="{status_class}">{check.status.value.upper()}</span></p>
    #             <p><strong>Check ID:</strong> {check.check_id}</p>
    #             <p><strong>Timestamp:</strong> {check.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    #         """
            
    #         if check.remediation_required and check.remediation_steps:
    #             html += "<p><strong>Remediation Steps:</strong></p><ul>"
    #             for step in check.remediation_steps:
    #                 html += f"<li>{step}</li>"
    #             html += "</ul>"
            
    #         html += "</div>"