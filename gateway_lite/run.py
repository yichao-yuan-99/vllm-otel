import uvicorn
from gateway_lite.app import create_app, create_gateway_service
from gateway_lite.port_profiles import load_port_profile
from gateway_lite.runtime_config import load_runtime_settings


def run():
    settings = load_runtime_settings()
    profile = load_port_profile(settings.port_profile_id)
    
    service = create_gateway_service()
    
    # Create regular gateway app (no response transformation)
    regular_app = create_app(service=service)
    
    # Create parsed gateway app (with response transformation)
    parsed_app = create_app(
        service=service,
        gateway_parse_port=profile.gateway_parse_port,
    )
    
    # This is a simplified runner - in production you'd want to run both
    # For now, just return the regular app for uvicorn
    return regular_app


if __name__ == "__main__":
    uvicorn.run(run(), host="0.0.0.0", port=11457)
