"""
Unit tests for Pandora AIOS FileSystem
"""

import unittest
from pandora_aios.filesystem import FileSystem, File


class TestFileSystem(unittest.TestCase):
    """Test cases for FileSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fs = FileSystem()
        
    def test_create_file(self):
        """Test file creation"""
        file = self.fs.create_file("/test.txt", "Hello World")
        self.assertEqual(file.name, "/test.txt")
        self.assertEqual(file.content, "Hello World")
        self.assertEqual(file.size, 11)
        
    def test_create_duplicate_file(self):
        """Test creating duplicate file raises error"""
        self.fs.create_file("/test.txt")
        with self.assertRaises(FileExistsError):
            self.fs.create_file("/test.txt")
            
    def test_read_file(self):
        """Test reading file content"""
        self.fs.create_file("/test.txt", "Test content")
        content = self.fs.read_file("/test.txt")
        self.assertEqual(content, "Test content")
        
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error"""
        with self.assertRaises(FileNotFoundError):
            self.fs.read_file("/nonexistent.txt")
            
    def test_write_file(self):
        """Test writing to file"""
        self.fs.create_file("/test.txt", "Initial")
        self.fs.write_file("/test.txt", "Updated")
        content = self.fs.read_file("/test.txt")
        self.assertEqual(content, "Updated")
        
    def test_delete_file(self):
        """Test deleting file"""
        self.fs.create_file("/test.txt")
        result = self.fs.delete_file("/test.txt")
        self.assertTrue(result)
        self.assertEqual(len(self.fs.list_files()), 0)
        
    def test_list_files(self):
        """Test listing files"""
        self.fs.create_file("/file1.txt")
        self.fs.create_file("/file2.txt")
        files = self.fs.list_files()
        self.assertEqual(len(files), 2)
        
    def test_get_total_size(self):
        """Test getting total file system size"""
        self.fs.create_file("/file1.txt", "abc")
        self.fs.create_file("/file2.txt", "defgh")
        total_size = self.fs.get_total_size()
        self.assertEqual(total_size, 8)


if __name__ == "__main__":
    unittest.main()
