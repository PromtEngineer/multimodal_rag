import json
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable
from flask import Response, request, jsonify
import traceback

logger = logging.getLogger(__name__)

# =============================================
# CORS Middleware
# =============================================

def add_cors_headers(response: Response) -> Response:
    """Add standardized CORS headers to response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

def cors_handler(app):
    """Configure CORS for Flask app"""
    @app.after_request
    def after_request(response):
        return add_cors_headers(response)
    
    @app.route('/<path:path>', methods=['OPTIONS'])
    @app.route('/', methods=['OPTIONS'])
    def handle_options(path=None):
        response = Response()
        return add_cors_headers(response)

# =============================================
# Error Handling Middleware
# =============================================

def create_error_response(error: str, status_code: int = 400, details: Dict[str, Any] = None) -> tuple:
    """Create standardized error response"""
    response_data = {
        "error": error,
        "status": "error",
        "status_code": status_code
    }
    
    if details:
        response_data["details"] = details
    
    logger.error(f"API Error {status_code}: {error}")
    if details:
        logger.error(f"Error details: {details}")
    
    return jsonify(response_data), status_code

def create_success_response(data: Any = None, message: str = "Success") -> Response:
    """Create standardized success response"""
    response_data = {
        "status": "success",
        "message": message
    }
    
    if data is not None:
        response_data["data"] = data
    
    return jsonify(response_data)

def error_handler(app):
    """Configure global error handlers for Flask app"""
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        return create_error_response("Bad request", 400)
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return create_error_response("Endpoint not found", 404)
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        return create_error_response("Method not allowed", 405)
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        logger.error(f"Internal server error: {error}")
        logger.error(traceback.format_exc())
        return create_error_response("Internal server error", 500)
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}")
        logger.error(traceback.format_exc())
        return create_error_response("An unexpected error occurred", 500)

# =============================================
# Request Validation Middleware
# =============================================

def require_json(f: Callable) -> Callable:
    """Decorator to require JSON content type"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return create_error_response("Content-Type must be application/json", 400)
        return f(*args, **kwargs)
    return decorated_function

def validate_fields(required_fields: list, optional_fields: list = None) -> Callable:
    """Decorator to validate required JSON fields"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                data = request.get_json()
                if not data:
                    return create_error_response("JSON data required", 400)
                
                # Check required fields
                missing_fields = []
                for field in required_fields:
                    if field not in data:
                        missing_fields.append(field)
                
                if missing_fields:
                    return create_error_response(
                        f"Missing required fields: {', '.join(missing_fields)}", 
                        400,
                        {"missing_fields": missing_fields}
                    )
                
                # Check for unexpected fields
                allowed_fields = set(required_fields + (optional_fields or []))
                unexpected_fields = set(data.keys()) - allowed_fields
                
                if unexpected_fields:
                    logger.warning(f"Unexpected fields in request: {unexpected_fields}")
                
                return f(*args, **kwargs)
            except json.JSONDecodeError:
                return create_error_response("Invalid JSON format", 400)
        return decorated_function
    return decorator

# =============================================
# Logging Middleware
# =============================================

def request_logger(app):
    """Configure request logging for Flask app"""
    
    @app.before_request
    def log_request():
        # Skip logging for health checks and static files
        if request.path in ['/health', '/favicon.ico']:
            return
        
        logger.info(f"🌐 {request.method} {request.path} from {request.remote_addr}")
        
        # Log request data for POST/PUT requests (but not file uploads)
        if request.method in ['POST', 'PUT'] and request.is_json:
            try:
                data = request.get_json()
                # Don't log sensitive data
                safe_data = {k: v for k, v in data.items() if k not in ['password', 'token', 'api_key']}
                if safe_data:
                    logger.debug(f"Request data: {safe_data}")
            except:
                pass  # Ignore JSON parsing errors for logging
    
    @app.after_request
    def log_response(response):
        if request.path not in ['/health', '/favicon.ico']:
            logger.info(f"📤 {request.method} {request.path} -> {response.status_code}")
        return response

# =============================================
# Health Check Utilities
# =============================================

def create_health_endpoint(app, additional_checks: Dict[str, Callable] = None):
    """Create a standardized health check endpoint"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        health_status = {
            "status": "healthy",
            "timestamp": str(datetime.now()),
            "service": app.name if hasattr(app, 'name') else "unknown",
            "checks": {}
        }
        
        # Run additional health checks if provided
        if additional_checks:
            for check_name, check_func in additional_checks.items():
                try:
                    check_result = check_func()
                    health_status["checks"][check_name] = {
                        "status": "pass" if check_result else "fail",
                        "result": check_result
                    }
                except Exception as e:
                    health_status["checks"][check_name] = {
                        "status": "fail",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] in ["healthy", "degraded"] else 503
        return jsonify(health_status), status_code

# =============================================
# Combined Middleware Setup
# =============================================

def setup_middleware(app, additional_health_checks: Dict[str, Callable] = None):
    """Setup all middleware for a Flask app"""
    cors_handler(app)
    error_handler(app)
    request_logger(app)
    create_health_endpoint(app, additional_health_checks)
    
    logger.info("✅ Middleware configured: CORS, Error Handling, Logging, Health Check")

# =============================================
# Response Helpers
# =============================================

def stream_response(generator, content_type: str = "text/plain"):
    """Create a streaming response"""
    return Response(
        generator,
        content_type=content_type,
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering for real-time streaming
        }
    )

def file_response(file_path: str, as_attachment: bool = False, attachment_filename: str = None):
    """Create a file download response"""
    from flask import send_file
    return send_file(
        file_path,
        as_attachment=as_attachment,
        download_name=attachment_filename
    )

# Import datetime for health check
from datetime import datetime 