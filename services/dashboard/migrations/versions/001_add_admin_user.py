"""
Adiciona usuário admin padrão.
"""
from datetime import datetime
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from ...core.security import get_password_hash

# Revisão identifiers
revision = '001_add_admin_user'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Adiciona usuário admin."""
    users = table('users',
        column('id', sa.Integer),
        column('username', sa.String),
        column('email', sa.String),
        column('hashed_password', sa.String),
        column('is_active', sa.Boolean),
        column('is_superuser', sa.Boolean),
        column('created_at', sa.DateTime)
    )
    
    # Senha padrão: Admin@123
    # Em produção, deve ser alterada imediatamente após o primeiro login
    op.bulk_insert(users, [
        {
            'username': 'admin',
            'email': 'admin@sistema.local',
            'hashed_password': get_password_hash('Admin@123'),
            'is_active': True,
            'is_superuser': True,
            'created_at': datetime.utcnow()
        }
    ])

def downgrade():
    """Remove usuário admin."""
    op.execute("DELETE FROM users WHERE username = 'admin'") 