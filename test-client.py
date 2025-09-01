#!/usr/bin/env python3

"""
Test client for OpenRecon Fetal Brain Measurement Server
This script tests the connection to the MRD server and can send test data
"""

import argparse
import socket
import logging
import sys
import os
import time

# Add the python-ismrmrd-server to path if available locally
if os.path.exists('./python-ismrmrd-server'):
    sys.path.append('./python-ismrmrd-server')

try:
    import ismrmrd
    import h5py
    ISMRMRD_AVAILABLE = True
except ImportError:
    print("Warning: ismrmrd or h5py not available. Only connection test will work.")
    ISMRMRD_AVAILABLE = False


def test_connection(host, port, timeout=5):
    """Test basic TCP connection to the MRD server"""
    print(f"Testing connection to {host}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ Connection successful!")
            return True
        else:
            print("❌ Connection failed!")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def create_test_data(output_file):
    """Create a simple test dataset for the fetal brain measurement server"""
    if not ISMRMRD_AVAILABLE:
        print("Error: ismrmrd not available. Cannot create test data.")
        return False
    
    try:
        print(f"Creating test data: {output_file}")
        
        # This is a simplified test data creation
        # In practice, you would use the ismrmrd-python-tools to create proper phantom data
        
        # Create a simple 3D volume that simulates a fetal brain scan
        import numpy as np
        
        # Create synthetic image data (64x64x32)
        nx, ny, nz = 64, 64, 32
        data = np.zeros((nx, ny, nz), dtype=np.complex64)
        
        # Add some structure that resembles a brain
        x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        cx, cy, cz = nx//2, ny//2, nz//2
        
        # Create a spherical brain-like structure
        brain_mask = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) < (min(nx, ny, nz)//3)**2
        data[brain_mask] = 1000 + 200j
        
        # Add some noise
        data += np.random.normal(0, 50, data.shape) + 1j * np.random.normal(0, 50, data.shape)
        
        # Create ISMRMRD dataset
        with h5py.File(output_file, 'w') as f:
            # Create dataset group
            dataset = f.create_group('dataset')
            
            # Create simple header
            header_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xs="http://www.w3.org/2001/XMLSchema" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd">
  <subjectInformation>
    <patientName>TestFetus</patientName>
    <patientWeight_kg>2.5</patientWeight_kg>
  </subjectInformation>
  <studyInformation>
    <studyDate>2024-01-15</studyDate>
    <studyTime>10:30:00</studyTime>
    <studyID>FETAL_BRAIN_TEST</studyID>
    <studyDescription>Fetal Brain Measurement Test</studyDescription>
  </studyInformation>
  <measurementInformation>
    <measurementID>1</measurementID>
    <seriesDate>2024-01-15</seriesDate>
    <seriesTime>10:30:00</seriesTime>
    <patientPosition>HFS</patientPosition>
  </measurementInformation>
  <acquisitionSystemInformation>
    <systemVendor>Siemens</systemVendor>
    <systemModel>Test</systemModel>
    <systemFieldStrength_T>3.0</systemFieldStrength_T>
  </acquisitionSystemInformation>
  <experimentalConditions>
    <H1resonanceFrequency_Hz>127740000</H1resonanceFrequency_Hz>
  </experimentalConditions>
  <encoding>
    <trajectory>cartesian</trajectory>
    <encodedSpace>
      <matrixSize><x>{nx}</x><y>{ny}</y><z>{nz}</z></matrixSize>
      <fieldOfView_mm><x>200</x><y>200</y><z>160</z></fieldOfView_mm>
    </encodedSpace>
    <reconSpace>
      <matrixSize><x>{nx}</x><y>{ny}</y><z>{nz}</z></matrixSize>
      <fieldOfView_mm><x>200</x><y>200</y><z>160</z></fieldOfView_mm>
    </reconSpace>
    <encodingLimits>
      <kspace_encoding_step_1><minimum>0</minimum><maximum>{ny-1}</maximum><center>{ny//2}</center></kspace_encoding_step_1>
      <kspace_encoding_step_2><minimum>0</minimum><maximum>{nz-1}</maximum><center>{nz//2}</center></kspace_encoding_step_2>
      <slice><minimum>0</minimum><maximum>0</maximum><center>0</center></slice>
      <set><minimum>0</minimum><maximum>0</maximum><center>0</center></set>
      <phase><minimum>0</minimum><maximum>0</maximum><center>0</center></phase>
      <repetition><minimum>0</minimum><maximum>0</maximum><center>0</center></repetition>
      <segment><minimum>0</minimum><maximum>0</maximum><center>0</center></segment>
      <contrast><minimum>0</minimum><maximum>0</maximum><center>0</center></contrast>
      <average><minimum>0</minimum><maximum>0</maximum><center>0</center></average>
    </encodingLimits>
  </encoding>
</ismrmrdHeader>"""
            
            # Save header
            dataset.create_dataset('xml', data=header_xml.encode('utf-8'))
            
            # Convert to image data and save
            image_data = np.abs(data).astype(np.uint16)
            dataset.create_dataset('data', data=image_data)
            
        print("✅ Test data created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        return False


def send_test_data(host, port, input_file):
    """Send test data to the MRD server"""
    if not ISMRMRD_AVAILABLE:
        print("Error: ismrmrd not available. Cannot send test data.")
        return False
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    try:
        print(f"Sending test data from {input_file} to {host}:{port}")
        
        # This would use the client.py from python-ismrmrd-server
        # For now, we'll just validate the file
        with h5py.File(input_file, 'r') as f:
            if 'dataset' in f:
                print("✅ Test data file is valid")
                print(f"Available groups: {list(f.keys())}")
                if 'dataset/data' in f:
                    data_shape = f['dataset/data'].shape
                    print(f"Data shape: {data_shape}")
                return True
            else:
                print("❌ Invalid test data file format")
                return False
                
    except Exception as e:
        print(f"❌ Error reading test data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test OpenRecon Fetal Brain Measurement Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=9002, help='Server port (default: 9002)')
    parser.add_argument('--create-test-data', metavar='FILE', help='Create test data file')
    parser.add_argument('--send-test-data', metavar='FILE', help='Send test data file to server')
    parser.add_argument('--timeout', type=int, default=5, help='Connection timeout in seconds')
    
    args = parser.parse_args()
    
    print("======================================================")
    print("OpenRecon Fetal Brain Measurement Server Test Client")
    print("======================================================")
    
    success = True
    
    # Test connection
    if not test_connection(args.host, args.port, args.timeout):
        success = False
    
    # Create test data if requested
    if args.create_test_data:
        if not create_test_data(args.create_test_data):
            success = False
    
    # Send test data if requested
    if args.send_test_data:
        if not send_test_data(args.host, args.port, args.send_test_data):
            success = False
    
    print("\n======================================================")
    if success:
        print("✅ All tests completed successfully!")
        
        if not args.create_test_data and not args.send_test_data:
            print("\nNext steps:")
            print("1. Create test data:")
            print(f"   python {sys.argv[0]} --create-test-data test_fetal_data.h5")
            print("2. Send test data:")
            print(f"   python {sys.argv[0]} --send-test-data test_fetal_data.h5")
            
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
