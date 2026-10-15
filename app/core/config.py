from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "loan-pledge-ocr-api"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    max_upload_mb: int = 15
    max_image_pixels: int = 40_000_000
    max_pdf_pages: int = 10

    ocr_lang: str = "en"
    ocr_use_angle_cls: bool = True
    ocr_preload: bool = True
    ocr_max_concurrency: int = 1
    ocr_queue_max: int = 6
    ocr_queue_timeout: float = 120
    ocr_max_side: int = 2500
    ocr_min_side: int = 1000
    ocr_denoise: bool = False
    ocr_refine: bool = True
    ocr_refine_max_crops: int = 40
    ocr_auto_rotate: bool = True
    ocr_deskew: bool = True
    trocr_model: str = "microsoft/trocr-small-handwritten"
    pdf_render_dpi: int = 200
    pdf_min_text_chars: int = 30
    field_min_ocr_confidence: float = 0.90
    field_min_match_confidence: float = 0.75
    mandatory_fields: str = "customerName,bankName,loanAccountNumber,pledgeNumber,loanAmount,loanDate,dueDate,branchName,phoneNumber,interestRate,tenure,netWeight,address"

    enable_trocr: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
    @property
    def mandatory_field_list(self) -> list[str]:
        return [f.strip() for f in self.mandatory_fields.split(",") if f.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()