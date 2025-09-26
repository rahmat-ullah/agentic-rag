"""Learning algorithms system for Sprint 6 Story 6-03

Revision ID: 007_learning_algorithms_system
Revises: 006_content_correction_system
Create Date: 2024-12-19 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '007_learning_algorithms_system'
down_revision = '006_content_correction_system'
branch_labels = None
depends_on = None


def upgrade():
    """Create learning algorithms system tables."""
    
    # Create learning_algorithms table
    op.create_table(
        'learning_algorithms',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('algorithm_type', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('learning_rate', sa.Float(), nullable=False, server_default='0.01'),
        sa.Column('validation_threshold', sa.Float(), nullable=False, server_default='0.05'),
        sa.Column('decay_factor', sa.Float(), nullable=True),
        sa.Column('regularization_strength', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('current_version', sa.String(20), nullable=False, server_default='1.0.0'),
        sa.Column('model_parameters', postgresql.JSONB(), nullable=True),
        sa.Column('training_data_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('precision_score', sa.Float(), nullable=True),
        sa.Column('recall_score', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('auto_update', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('validation_frequency_hours', sa.Integer(), nullable=False, server_default='24'),
        sa.Column('algorithm_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_trained_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_validated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.CheckConstraint('learning_rate > 0.0 AND learning_rate <= 1.0', name='check_learning_rate_range'),
        sa.CheckConstraint('validation_threshold >= 0.0 AND validation_threshold <= 1.0', name='check_validation_threshold_range'),
        sa.CheckConstraint('accuracy_score IS NULL OR accuracy_score BETWEEN 0.0 AND 1.0', name='check_accuracy_score_range'),
        sa.CheckConstraint('precision_score IS NULL OR precision_score BETWEEN 0.0 AND 1.0', name='check_precision_score_range'),
        sa.CheckConstraint('recall_score IS NULL OR recall_score BETWEEN 0.0 AND 1.0', name='check_recall_score_range'),
        sa.CheckConstraint('f1_score IS NULL OR f1_score BETWEEN 0.0 AND 1.0', name='check_f1_score_range'),
    )
    
    # Create learning_sessions table
    op.create_table(
        'learning_sessions',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('algorithm_id', postgresql.UUID(), nullable=False),
        sa.Column('session_type', sa.String(50), nullable=False),
        sa.Column('session_name', sa.String(100), nullable=True),
        sa.Column('input_data_size', sa.Integer(), nullable=False),
        sa.Column('processed_records', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('successful_updates', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_updates', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('cpu_usage_percent', sa.Float(), nullable=True),
        sa.Column('performance_improvement', sa.Float(), nullable=True),
        sa.Column('accuracy_change', sa.Float(), nullable=True),
        sa.Column('error_rate', sa.Float(), nullable=True),
        sa.Column('session_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('error_details', postgresql.JSONB(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['algorithm_id'], ['learning_algorithms.id']),
    )
    
    # Create learning_performance_metrics table
    op.create_table(
        'learning_performance_metrics',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('algorithm_id', postgresql.UUID(), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('measurement_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('measurement_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('baseline_value', sa.Float(), nullable=True),
        sa.Column('improvement_percentage', sa.Float(), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('metric_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['algorithm_id'], ['learning_algorithms.id']),
    )
    
    # Create feedback_signals table
    op.create_table(
        'feedback_signals',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('signal_type', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('target_id', postgresql.UUID(), nullable=False),
        sa.Column('signal_value', sa.Float(), nullable=False),
        sa.Column('signal_strength', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('signal_confidence', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('user_id', postgresql.UUID(), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=True),
        sa.Column('query_context', sa.Text(), nullable=True),
        sa.Column('is_processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_algorithm_id', postgresql.UUID(), nullable=True),
        sa.Column('signal_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['user_id'], ['app_user.id']),
        sa.ForeignKeyConstraint(['processing_algorithm_id'], ['learning_algorithms.id']),
        sa.CheckConstraint('signal_strength >= 0.0 AND signal_strength <= 1.0', name='check_signal_strength_range'),
        sa.CheckConstraint('signal_confidence >= 0.0 AND signal_confidence <= 1.0', name='check_signal_confidence_range'),
    )
    
    # Create learning_model_states table
    op.create_table(
        'learning_model_states',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('algorithm_id', postgresql.UUID(), nullable=False),
        sa.Column('state_name', sa.String(100), nullable=False),
        sa.Column('state_version', sa.String(20), nullable=False),
        sa.Column('state_type', sa.String(50), nullable=False),
        sa.Column('state_data', postgresql.JSONB(), nullable=False),
        sa.Column('state_size_bytes', sa.Integer(), nullable=True),
        sa.Column('compression_type', sa.String(20), nullable=True),
        sa.Column('is_validated', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('validation_score', sa.Float(), nullable=True),
        sa.Column('validation_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_backup', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('parent_state_id', postgresql.UUID(), nullable=True),
        sa.Column('state_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['algorithm_id'], ['learning_algorithms.id']),
        sa.ForeignKeyConstraint(['parent_state_id'], ['learning_model_states.id']),
        sa.CheckConstraint('validation_score IS NULL OR validation_score BETWEEN 0.0 AND 1.0', name='check_validation_score_range'),
    )
    
    # Create ab_test_experiments table
    op.create_table(
        'ab_test_experiments',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(), nullable=False),
        sa.Column('experiment_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('hypothesis', sa.Text(), nullable=True),
        sa.Column('control_algorithm_id', postgresql.UUID(), nullable=False),
        sa.Column('treatment_algorithm_id', postgresql.UUID(), nullable=False),
        sa.Column('traffic_split_percentage', sa.Float(), nullable=False, server_default='50.0'),
        sa.Column('status', sa.String(20), nullable=False, server_default='draft'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('primary_metric', sa.String(100), nullable=False),
        sa.Column('success_threshold', sa.Float(), nullable=False),
        sa.Column('minimum_sample_size', sa.Integer(), nullable=False, server_default='1000'),
        sa.Column('confidence_level', sa.Float(), nullable=False, server_default='0.95'),
        sa.Column('control_metric_value', sa.Float(), nullable=True),
        sa.Column('treatment_metric_value', sa.Float(), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('effect_size', sa.Float(), nullable=True),
        sa.Column('experiment_metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['control_algorithm_id'], ['learning_algorithms.id']),
        sa.ForeignKeyConstraint(['treatment_algorithm_id'], ['learning_algorithms.id']),
        sa.CheckConstraint('traffic_split_percentage > 0.0 AND traffic_split_percentage <= 100.0', name='check_traffic_split_range'),
        sa.CheckConstraint('confidence_level > 0.0 AND confidence_level < 1.0', name='check_confidence_level_range'),
        sa.CheckConstraint('statistical_significance IS NULL OR statistical_significance BETWEEN 0.0 AND 1.0', name='check_statistical_significance_range'),
    )
    
    # Create indexes for learning_algorithms table
    op.create_index('idx_learning_algorithms_tenant_type', 'learning_algorithms', ['tenant_id', 'algorithm_type'])
    op.create_index('idx_learning_algorithms_status', 'learning_algorithms', ['status'])
    op.create_index('idx_learning_algorithms_enabled', 'learning_algorithms', ['is_enabled'])
    op.create_index('idx_learning_algorithms_updated', 'learning_algorithms', ['updated_at'])
    
    # Create indexes for learning_sessions table
    op.create_index('idx_learning_sessions_algorithm', 'learning_sessions', ['algorithm_id'])
    op.create_index('idx_learning_sessions_type', 'learning_sessions', ['session_type'])
    op.create_index('idx_learning_sessions_started', 'learning_sessions', ['started_at'])
    op.create_index('idx_learning_sessions_completed', 'learning_sessions', ['completed_at'])
    
    # Create indexes for learning_performance_metrics table
    op.create_index('idx_learning_performance_algorithm', 'learning_performance_metrics', ['algorithm_id'])
    op.create_index('idx_learning_performance_metric', 'learning_performance_metrics', ['metric_name'])
    op.create_index('idx_learning_performance_recorded', 'learning_performance_metrics', ['recorded_at'])
    op.create_index('idx_learning_performance_period', 'learning_performance_metrics', ['measurement_period_start', 'measurement_period_end'])
    
    # Create indexes for feedback_signals table
    op.create_index('idx_feedback_signals_tenant_type', 'feedback_signals', ['tenant_id', 'signal_type'])
    op.create_index('idx_feedback_signals_target', 'feedback_signals', ['target_type', 'target_id'])
    op.create_index('idx_feedback_signals_processed', 'feedback_signals', ['is_processed'])
    op.create_index('idx_feedback_signals_created', 'feedback_signals', ['created_at'])
    op.create_index('idx_feedback_signals_user', 'feedback_signals', ['user_id'])
    
    # Create indexes for learning_model_states table
    op.create_index('idx_learning_model_states_algorithm', 'learning_model_states', ['algorithm_id'])
    op.create_index('idx_learning_model_states_name_version', 'learning_model_states', ['state_name', 'state_version'])
    op.create_index('idx_learning_model_states_active', 'learning_model_states', ['is_active'])
    op.create_index('idx_learning_model_states_validated', 'learning_model_states', ['is_validated'])
    op.create_index('idx_learning_model_states_created', 'learning_model_states', ['created_at'])
    
    # Create indexes for ab_test_experiments table
    op.create_index('idx_ab_test_experiments_tenant', 'ab_test_experiments', ['tenant_id'])
    op.create_index('idx_ab_test_experiments_status', 'ab_test_experiments', ['status'])
    op.create_index('idx_ab_test_experiments_active', 'ab_test_experiments', ['is_active'])
    op.create_index('idx_ab_test_experiments_started', 'ab_test_experiments', ['started_at'])
    
    # Add check constraints for enum values
    op.create_check_constraint(
        'ck_algorithm_type',
        'learning_algorithms',
        "algorithm_type IN ('link_confidence', 'chunk_ranking', 'query_expansion', 'negative_feedback', 'content_quality')"
    )
    
    op.create_check_constraint(
        'ck_model_type',
        'learning_algorithms',
        "model_type IN ('exponential_moving_average', 'bayesian_update', 'reinforcement_learning', 'collaborative_filtering', 'content_based_filtering', 'neural_language_model')"
    )
    
    op.create_check_constraint(
        'ck_learning_status',
        'learning_algorithms',
        "status IN ('active', 'paused', 'training', 'validating', 'disabled')"
    )
    
    op.create_check_constraint(
        'ck_signal_type',
        'feedback_signals',
        "signal_type IN ('click_through', 'dwell_time', 'explicit_rating', 'bounce_rate', 'conversion_rate', 'correction_feedback')"
    )
    
    op.create_check_constraint(
        'ck_experiment_status',
        'ab_test_experiments',
        "status IN ('draft', 'running', 'paused', 'completed', 'cancelled')"
    )


def downgrade():
    """Drop learning algorithms system tables."""
    
    # Drop indexes first
    op.drop_index('idx_ab_test_experiments_started', 'ab_test_experiments')
    op.drop_index('idx_ab_test_experiments_active', 'ab_test_experiments')
    op.drop_index('idx_ab_test_experiments_status', 'ab_test_experiments')
    op.drop_index('idx_ab_test_experiments_tenant', 'ab_test_experiments')
    
    op.drop_index('idx_learning_model_states_created', 'learning_model_states')
    op.drop_index('idx_learning_model_states_validated', 'learning_model_states')
    op.drop_index('idx_learning_model_states_active', 'learning_model_states')
    op.drop_index('idx_learning_model_states_name_version', 'learning_model_states')
    op.drop_index('idx_learning_model_states_algorithm', 'learning_model_states')
    
    op.drop_index('idx_feedback_signals_user', 'feedback_signals')
    op.drop_index('idx_feedback_signals_created', 'feedback_signals')
    op.drop_index('idx_feedback_signals_processed', 'feedback_signals')
    op.drop_index('idx_feedback_signals_target', 'feedback_signals')
    op.drop_index('idx_feedback_signals_tenant_type', 'feedback_signals')
    
    op.drop_index('idx_learning_performance_period', 'learning_performance_metrics')
    op.drop_index('idx_learning_performance_recorded', 'learning_performance_metrics')
    op.drop_index('idx_learning_performance_metric', 'learning_performance_metrics')
    op.drop_index('idx_learning_performance_algorithm', 'learning_performance_metrics')
    
    op.drop_index('idx_learning_sessions_completed', 'learning_sessions')
    op.drop_index('idx_learning_sessions_started', 'learning_sessions')
    op.drop_index('idx_learning_sessions_type', 'learning_sessions')
    op.drop_index('idx_learning_sessions_algorithm', 'learning_sessions')
    
    op.drop_index('idx_learning_algorithms_updated', 'learning_algorithms')
    op.drop_index('idx_learning_algorithms_enabled', 'learning_algorithms')
    op.drop_index('idx_learning_algorithms_status', 'learning_algorithms')
    op.drop_index('idx_learning_algorithms_tenant_type', 'learning_algorithms')
    
    # Drop tables
    op.drop_table('ab_test_experiments')
    op.drop_table('learning_model_states')
    op.drop_table('feedback_signals')
    op.drop_table('learning_performance_metrics')
    op.drop_table('learning_sessions')
    op.drop_table('learning_algorithms')
