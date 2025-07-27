"""
File Operations Tool for BabyAGI
Provides safe file reading, writing, and manipulation capabilities.
"""

import os
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Union
from .base_tool import BaseTool, ToolResult, ToolStatus


class FileOperationsTool(BaseTool):
    """Tool for file operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="file_operations",
            description="Read, write, and manipulate files safely",
            config=config
        )
        
        # Security settings
        self.allowed_extensions = config.get("allowed_extensions", [
            '.txt', '.json', '.csv', '.md', '.py', '.html', '.xml', '.yaml', '.yml'
        ])
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.sandbox_dir = config.get("sandbox_dir", "./sandbox")
        
        # Create sandbox directory
        os.makedirs(self.sandbox_dir, exist_ok=True)
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute file operation."""
        start_time = time.time()
        
        operations = {
            "read": self._read_file,
            "write": self._write_file,
            "list": self._list_files,
            "delete": self._delete_file,
            "exists": self._file_exists,
            "info": self._file_info,
            "create_dir": self._create_directory,
            "read_csv": self._read_csv,
            "write_csv": self._write_csv,
            "read_json": self._read_json,
            "write_json": self._write_json
        }
        
        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Unknown operation: {operation}"
            )
        
        try:
            result = operations[operation](**kwargs)
            execution_time = time.time() - start_time
            
            self._update_stats(True)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"File operation '{operation}' completed successfully",
                metadata={"operation": operation},
                execution_time=execution_time
            )
            
        except Exception as e:
            self._update_stats(False)
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"File operation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _validate_path(self, filepath: str) -> str:
        """Validate and sanitize file path."""
        # Convert to Path object for safe handling
        path = Path(filepath)
        
        # Ensure path is within sandbox
        if not path.is_absolute():
            path = Path(self.sandbox_dir) / path
        
        # Resolve to absolute path and check if it's within sandbox
        resolved_path = path.resolve()
        sandbox_path = Path(self.sandbox_dir).resolve()
        
        if not str(resolved_path).startswith(str(sandbox_path)):
            raise ValueError(f"Path outside sandbox directory: {filepath}")
        
        return str(resolved_path)
    
    def _check_file_extension(self, filepath: str) -> bool:
        """Check if file extension is allowed."""
        ext = Path(filepath).suffix.lower()
        return ext in self.allowed_extensions
    
    def _read_file(self, filepath: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file content."""
        validated_path = self._validate_path(filepath)
        
        if not os.path.exists(validated_path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not self._check_file_extension(validated_path):
            raise ValueError(f"File extension not allowed: {Path(filepath).suffix}")
        
        file_size = os.path.getsize(validated_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        with open(validated_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "content": content,
            "size": file_size,
            "encoding": encoding,
            "path": filepath
        }
    
    def _write_file(self, filepath: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write content to file."""
        validated_path = self._validate_path(filepath)
        
        if not self._check_file_extension(validated_path):
            raise ValueError(f"File extension not allowed: {Path(filepath).suffix}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(validated_path), exist_ok=True)
        
        with open(validated_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return {
            "bytes_written": len(content.encode(encoding)),
            "encoding": encoding,
            "path": filepath
        }
    
    def _list_files(self, directory: str = ".", pattern: str = "*") -> Dict[str, Any]:
        """List files in directory."""
        validated_path = self._validate_path(directory)
        
        if not os.path.exists(validated_path):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not os.path.isdir(validated_path):
            raise ValueError(f"Path is not a directory: {directory}")
        
        files = []
        directories = []
        
        for item in Path(validated_path).glob(pattern):
            if item.is_file():
                files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "modified": item.stat().st_mtime,
                    "extension": item.suffix
                })
            elif item.is_dir():
                directories.append({
                    "name": item.name,
                    "modified": item.stat().st_mtime
                })
        
        return {
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_directories": len(directories)
        }
    
    def _delete_file(self, filepath: str) -> Dict[str, Any]:
        """Delete file."""
        validated_path = self._validate_path(filepath)
        
        if not os.path.exists(validated_path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_size = os.path.getsize(validated_path)
        os.remove(validated_path)
        
        return {
            "deleted": True,
            "size_freed": file_size,
            "path": filepath
        }
    
    def _file_exists(self, filepath: str) -> Dict[str, Any]:
        """Check if file exists."""
        validated_path = self._validate_path(filepath)
        exists = os.path.exists(validated_path)
        
        result = {"exists": exists, "path": filepath}
        
        if exists:
            stat = os.stat(validated_path)
            result.update({
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_file": os.path.isfile(validated_path),
                "is_directory": os.path.isdir(validated_path)
            })
        
        return result
    
    def _file_info(self, filepath: str) -> Dict[str, Any]:
        """Get detailed file information."""
        validated_path = self._validate_path(filepath)
        
        if not os.path.exists(validated_path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = os.stat(validated_path)
        path_obj = Path(validated_path)
        
        return {
            "name": path_obj.name,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "extension": path_obj.suffix,
            "is_file": path_obj.is_file(),
            "is_directory": path_obj.is_dir(),
            "permissions": oct(stat.st_mode)[-3:],
            "path": filepath
        }
    
    def _create_directory(self, directory: str) -> Dict[str, Any]:
        """Create directory."""
        validated_path = self._validate_path(directory)
        os.makedirs(validated_path, exist_ok=True)
        
        return {
            "created": True,
            "path": directory
        }
    
    def _read_csv(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Read CSV file."""
        validated_path = self._validate_path(filepath)
        
        if not validated_path.endswith('.csv'):
            raise ValueError("File must have .csv extension")
        
        rows = []
        with open(validated_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, **kwargs)
            for row in reader:
                rows.append(dict(row))
        
        return {
            "data": rows,
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "path": filepath
        }
    
    def _write_csv(self, filepath: str, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Write CSV file."""
        validated_path = self._validate_path(filepath)
        
        if not validated_path.endswith('.csv'):
            raise ValueError("File must have .csv extension")
        
        if not data:
            raise ValueError("No data provided")
        
        fieldnames = list(data[0].keys())
        
        os.makedirs(os.path.dirname(validated_path), exist_ok=True)
        
        with open(validated_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(data)
        
        return {
            "rows_written": len(data),
            "columns": fieldnames,
            "path": filepath
        }
    
    def _read_json(self, filepath: str) -> Dict[str, Any]:
        """Read JSON file."""
        validated_path = self._validate_path(filepath)
        
        if not validated_path.endswith('.json'):
            raise ValueError("File must have .json extension")
        
        with open(validated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "data": data,
            "path": filepath
        }
    
    def _write_json(self, filepath: str, data: Any, indent: int = 2) -> Dict[str, Any]:
        """Write JSON file."""
        validated_path = self._validate_path(filepath)
        
        if not validated_path.endswith('.json'):
            raise ValueError("File must have .json extension")
        
        os.makedirs(os.path.dirname(validated_path), exist_ok=True)
        
        with open(validated_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return {
            "path": filepath,
            "size": os.path.getsize(validated_path)
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "File operation to perform",
                    "enum": ["read", "write", "list", "delete", "exists", "info", 
                            "create_dir", "read_csv", "write_csv", "read_json", "write_json"]
                }
            }
        }

