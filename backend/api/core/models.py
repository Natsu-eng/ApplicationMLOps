"""Modèles ORM — Organisation (bureau d'études) et utilisateurs.

Modèle multi-tenant retenu (voir le diagnostic de migration, section E) :
une Organisation = un bureau d'études, plusieurs utilisateurs y appartiennent
(rôle "owner" ou "member"). Toute donnée métier créée dans les lots suivants
(datasets, jobs d'entraînement, modèles) portera un organization_id et ne
sera jamais visible en dehors de son organisation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.core.database import Base


class Organization(Base):
    """Un bureau d'études — unité d'isolation des données."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship("User", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    nom: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    # "owner" : a créé l'organisation, seul rôle autorisé à inviter des membres (Lot 1).
    # "member" : rattaché à une organisation existante par un owner.
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="member")
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actif: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    @property
    def organization_name(self) -> str:
        """Exposé pour les réponses API — évite de sérialiser l'objet Organization entier."""
        return self.organization.name


class Dataset(Base):
    """Un jeu de données tabulaire uploadé — appartient à l'organisation entière
    (pas seulement à qui l'a uploadé), cohérent avec le principe d'équipe partagée."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    uploaded_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # JSON list [{"name": ..., "dtype": ...}, ...] — évite une table séparée pour Lot 2
    columns_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")  # processing | ready | error
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    uploaded_by: Mapped[Optional["User"]] = relationship("User")
