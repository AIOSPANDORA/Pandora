"""
Unit tests for Internet Consciousness Stream
"""

import unittest
import time
import os
from internet_consciousness_stream import (
    InternetConsciousnessStream,
    DataSource,
    ConsciousnessStreamConfig,
    initiate_stream,
    get_stream
)


class TestDataSource(unittest.TestCase):
    """Test cases for DataSource"""
    
    def test_data_source_creation(self):
        """Test creating a DataSource"""
        source = DataSource(
            name="Test Source",
            url="http://example.com/rss",
            source_type="rss",
            enabled=True,
            fetch_interval=300
        )
        
        self.assertEqual(source.name, "Test Source")
        self.assertEqual(source.url, "http://example.com/rss")
        self.assertEqual(source.source_type, "rss")
        self.assertTrue(source.enabled)
        self.assertEqual(source.fetch_interval, 300)
        self.assertIsNone(source.last_fetch)
    
    def test_data_source_to_dict(self):
        """Test DataSource serialization"""
        source = DataSource(
            name="Test",
            url="http://test.com",
            source_type="api"
        )
        
        data = source.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "Test")
        self.assertEqual(data['url'], "http://test.com")
        self.assertEqual(data['type'], "api")


class TestConsciousnessStreamConfig(unittest.TestCase):
    """Test cases for ConsciousnessStreamConfig"""
    
    def test_load_default_sources(self):
        """Test loading default data sources"""
        sources = ConsciousnessStreamConfig.load_sources()
        
        self.assertIsInstance(sources, list)
        self.assertGreater(len(sources), 0)
        
        # Check that all sources are DataSource instances
        for source in sources:
            self.assertIsInstance(source, DataSource)
    
    def test_save_and_load_sources(self):
        """Test saving and loading sources from file"""
        test_file = "/tmp/test_consciousness_sources.json"
        
        # Create test sources
        test_sources = [
            DataSource(
                name="Test Source 1",
                url="http://test1.com",
                source_type="rss"
            ),
            DataSource(
                name="Test Source 2",
                url="http://test2.com",
                source_type="api",
                enabled=False
            )
        ]
        
        # Save
        ConsciousnessStreamConfig.save_sources(test_sources, test_file)
        self.assertTrue(os.path.exists(test_file))
        
        # Load
        loaded_sources = ConsciousnessStreamConfig.load_sources(test_file)
        self.assertEqual(len(loaded_sources), 2)
        self.assertEqual(loaded_sources[0].name, "Test Source 1")
        self.assertEqual(loaded_sources[1].name, "Test Source 2")
        self.assertFalse(loaded_sources[1].enabled)
        
        # Cleanup
        os.remove(test_file)


class TestInternetConsciousnessStream(unittest.TestCase):
    """Test cases for InternetConsciousnessStream"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config_file = "/tmp/test_consciousness_config.json"
        self.stream = InternetConsciousnessStream(config_file=self.test_config_file)
    
    def tearDown(self):
        """Clean up after tests"""
        if self.stream.running:
            self.stream.stop_stream()
        
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_stream_initialization(self):
        """Test stream initialization"""
        self.assertIsNotNone(self.stream)
        self.assertFalse(self.stream.running)
        self.assertIsInstance(self.stream.sources, list)
        self.assertGreater(len(self.stream.sources), 0)
    
    def test_start_stop_stream(self):
        """Test starting and stopping the stream"""
        self.assertFalse(self.stream.running)
        
        # Start stream
        self.stream.start_stream()
        self.assertTrue(self.stream.running)
        self.assertIsNotNone(self.stream.thread)
        
        # Give it a moment to start
        time.sleep(0.5)
        
        # Stop stream
        self.stream.stop_stream()
        self.assertFalse(self.stream.running)
    
    def test_add_source(self):
        """Test adding a new source"""
        initial_count = len(self.stream.sources)
        
        new_source = DataSource(
            name="New Test Source",
            url="http://newtest.com",
            source_type="rss"
        )
        
        self.stream.add_source(new_source)
        
        self.assertEqual(len(self.stream.sources), initial_count + 1)
        self.assertEqual(self.stream.sources[-1].name, "New Test Source")
    
    def test_remove_source(self):
        """Test removing a source"""
        # Add a source
        test_source = DataSource(
            name="Removable Source",
            url="http://removable.com",
            source_type="api"
        )
        self.stream.add_source(test_source)
        
        initial_count = len(self.stream.sources)
        
        # Remove it
        self.stream.remove_source("Removable Source")
        
        self.assertEqual(len(self.stream.sources), initial_count - 1)
        
        # Verify it's gone
        source_names = [s.name for s in self.stream.sources]
        self.assertNotIn("Removable Source", source_names)
    
    def test_get_stats(self):
        """Test getting statistics"""
        stats = self.stream.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('running', stats)
        self.assertIn('sources_count', stats)
        self.assertIn('enabled_sources', stats)
        self.assertIn('total_fetches', stats)
        self.assertIn('successful_fetches', stats)
        self.assertIn('failed_fetches', stats)
        self.assertIn('data_points_collected', stats)
        
        self.assertEqual(stats['running'], False)
        self.assertGreaterEqual(stats['sources_count'], 0)
    
    def test_get_sources(self):
        """Test getting sources list"""
        sources = self.stream.get_sources()
        
        self.assertIsInstance(sources, list)
        
        for source in sources:
            self.assertIsInstance(source, dict)
            self.assertIn('name', source)
            self.assertIn('url', source)
            self.assertIn('type', source)
            self.assertIn('enabled', source)


class TestGlobalStreamFunctions(unittest.TestCase):
    """Test cases for global stream functions"""
    
    def setUp(self):
        """Reset global stream instance before each test"""
        import internet_consciousness_stream
        internet_consciousness_stream._stream_instance = None
    
    def tearDown(self):
        """Clean up global stream after each test"""
        import internet_consciousness_stream
        if internet_consciousness_stream._stream_instance:
            internet_consciousness_stream._stream_instance.stop_stream()
            internet_consciousness_stream._stream_instance = None
    
    def test_initiate_stream(self):
        """Test initiating the global stream"""
        stream = initiate_stream("/tmp/test_global_stream.json")
        
        self.assertIsNotNone(stream)
        self.assertIsInstance(stream, InternetConsciousnessStream)
        self.assertTrue(stream.running)
        
        # Cleanup
        if os.path.exists("/tmp/test_global_stream.json"):
            os.remove("/tmp/test_global_stream.json")
    
    def test_get_stream(self):
        """Test getting the global stream instance"""
        # First initiate
        stream1 = initiate_stream("/tmp/test_get_stream.json")
        
        # Then get
        stream2 = get_stream()
        
        # Should be the same instance
        self.assertIs(stream1, stream2)
        
        # Cleanup
        if os.path.exists("/tmp/test_get_stream.json"):
            os.remove("/tmp/test_get_stream.json")


if __name__ == '__main__':
    unittest.main()
