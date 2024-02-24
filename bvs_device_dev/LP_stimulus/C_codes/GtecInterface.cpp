
#include "GtecInterface.h"

GtecInterface::GtecInterface()
{
	Status = gcnew ResultStruct();
	//Initialize Private Variables
	//Select_Channels = gcnew array <int ^> {1,1,1,1,1,1,1,1}; // {Fz, Cz, P3, Pz, P4, P07, P08, Oz}  Setting some of these to 0 will result in no data collected for these channels
	Electrode_Names = gcnew array <String ^> (GDS_GNAUTILUS_CHANNELS_MAX);
	Device_Name = nullptr;
	RunData = nullptr;
//	RunData->Duration = DURATION_DAQ; //Default - 310 sec
	NumChannels = NULL;
}

void GtecInterface::FindDevices(void)
{	
	//
	try
	{	
		//-------------------------------------------------------------------------------------
		// Initialize the library.
		//-------------------------------------------------------------------------------------

		GDS_Initialize();
		
		Status->Success = true;
		Status->Message = "";

		Device_Name = nullptr;


		//-------------------------------------------------------------------------------------
		// Setup network addresses.
		//-------------------------------------------------------------------------------------
		static const std::string host_ip(HOST_IP);
		static const uint16_t host_port = HOST_PORT;
		static const std::string local_ip(LOCAL_IP);
		static const uint16_t local_port = LOCAL_PORT;

		GDS_ENDPOINT local_endpoint, host_endpoint;
		strcpy( host_endpoint.IpAddress, host_ip.c_str() );
		host_endpoint.Port = host_port;
		strcpy( local_endpoint.IpAddress, local_ip.c_str() );
		local_endpoint.Port = local_port;

		//-------------------------------------------------------------------------------------
		// Identify Connected Device
		//-------------------------------------------------------------------------------------
		GDS_DEVICE_CONNECTION_INFO* connected_devices = NULL;
		size_t count_daq_units = 0;

		GDS_RESULT ret = GDS_GetConnectedDevices( host_endpoint, local_endpoint, &connected_devices, &count_daq_units );
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		if (count_daq_units > 0)
		{

			System::Windows::Forms::DialogResult SelectDeviceResult;
			NeuroCatch::SelectDevice^ SelectDeviceForm = gcnew NeuroCatch::SelectDevice();

			String^ temp;
			for (size_t i = 0; i < count_daq_units ; i++)
			{
				temp = gcnew String(connected_devices[i].ConnectedDevices[0].Name);
				SelectDeviceForm->DeviceListBox->Items->Add(temp);
			}

			SelectDeviceResult = SelectDeviceForm->ShowDialog();
			while ( SelectDeviceResult!= System::Windows::Forms::DialogResult::OK )
			{
			}

			size_t index = SelectDeviceForm->DeviceListBox->SelectedIndex;
			Device_Name = SelectDeviceForm->DeviceListBox->SelectedItem->ToString();

			device = new char[1][DEVICE_NAME_LENGTH_MAX];
			std::strcpy( device[0], connected_devices[index].ConnectedDevices[0].Name); //device[0] is thus a character array containing the name of the connected device

			delete SelectDeviceForm;

			if ( connected_devices[index].ConnectedDevices[0].DeviceType != GDS_DEVICE_TYPE_GNAUTILUS )
			{
				throw gcnew Exception(String::Format("The selected device ({0}) is not a gNautilus\n", Device_Name)); 
			}

			else if (connected_devices[index].InUse) //i.e. go ahead and connect
			{
				throw gcnew Exception(String::Format("The selected device ({0}) is currently in use. Please disconnect and try again.\n", Device_Name));
			}
		}
		else {//No Connected Devices
			throw gcnew Exception("No connected devices have been found. Please connect a device and try again\n"); }

		//-------------------------------------------------------------------------------------
		// Free the list of found devices. This list is not needed any more.
		//-------------------------------------------------------------------------------------
		ret = GDS_FreeConnectedDevicesList( &connected_devices, count_daq_units);
		if ( ret.ErrorCode )
			throw gcnew Exception(gcnew String(ret.ErrorMessage));


		//Connect to Device

		BOOL is_creator = FALSE;
		ret = GDS_Connect( host_endpoint, local_endpoint, device, 1, TRUE, &connectionHandle, &is_creator );

		if ( ret.ErrorCode ) {
		throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Function to Set the callback functions in case the server triggers a DAQ error event
		//-------------------------------------------------------------------------------------	
		std::string usrData;
		String^ temp = Device_Name;
		usrData = msclr::interop::marshal_as<std::string>( temp );

		ret = GDS_SetDataAcquisitionErrorCallback(connectionHandle, on_data_acquisition_error, (void*) &usrData);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Set the callback function which will be called if the server dies or is going to
		// shutdown.
		//-------------------------------------------------------------------------------------

		ret = GDS_SetServerDiedCallback(connectionHandle, on_server_died_event, (void*) &usrData);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }


		//-------------------------------------------------------------------------------------
		// SF-  Get and save number and names of Electrodes
		//-------------------------------------------------------------------------------------
		uint32_t mountedModulesCount = NULL;
		size_t electrodeCount = NULL;
		char (*electrodeNames)[GDS_GNAUTILUS_ELECTRODE_NAME_LENGTH_MAX] = NULL; //find number of modules and number of electrodes
		ret = GDS_GNAUTILUS_GetChannelNames(connectionHandle, device, &mountedModulesCount, electrodeNames, &electrodeCount);

		electrodeNames = new char[electrodeCount][GDS_GNAUTILUS_ELECTRODE_NAME_LENGTH_MAX]; //re-initialise with prior knowledge
		ret = GDS_GNAUTILUS_GetChannelNames(connectionHandle, device, &mountedModulesCount, electrodeNames, &electrodeCount);

		//std::cout << "There are " << mountedModulesCount << " available Modules:" << std::endl;
		//std::cout << "There are " << electrodeCount << " available electrodes:" << std::endl;
		for ( size_t i = 0; i < electrodeCount; i++ )
		{
			Electrode_Names[i] = gcnew String(electrodeNames[i]);
		}

		NumChannels = electrodeCount;
		//-------------------------------------------------------------------------------------
		// Setup Server Configuration
		//-------------------------------------------------------------------------------------
		GDS_GNAUTILUS_CONFIGURATION* config = new GDS_GNAUTILUS_CONFIGURATION;

		config->Slave = FALSE;
		config->SamplingRate = SAMPLE_RATE;	
		config->NumberOfScans = 0; //Automatic
		config->NetworkChannel = 0; //Automatic
		config->DigitalIOs = TRUE;
		config->InputSignal = GDS_GNAUTILUS_INPUT_SIGNAL_ELECTRODE;
		config->AccelerationData = TRUE;
		config->BatteryLevel = TRUE;
		config->CAR = FALSE;
		config->Counter = TRUE;
		config->LinkQualityInformation = FALSE;
		config->ValidationIndicator = FALSE;
		config->NoiseReduction = FALSE;

		for (uint32_t j = 0; j < GDS_GNAUTILUS_CHANNELS_MAX; j++)
		{
			if (j < NumChannels)  
				config->Channels[j].Enabled = TRUE;
			else 
				config->Channels[j].Enabled = FALSE;
			
			config->Channels[j].Sensitivity = 187500.0;
			config->Channels[j].BandpassFilterIndex = GDS_GNAUTILUS_BANDPASS_FILTER_3; //0.1 - 100Hz, 6th Order Butterworth @ 500Hz
			config->Channels[j].NotchFilterIndex = GDS_GNAUTILUS_NOTCH_FILTER_60Hz_500;
			config->Channels[j].BipolarChannel = GDS_GNAUTILUS_NO_BIPOLAR_CHANNEL_IDX;	
			config->Channels[j].UsedForNoiseReduction = FALSE;
			config->Channels[j].UsedForCar = FALSE;
		}

		GDS_CONFIGURATION_BASE* cfg = new GDS_CONFIGURATION_BASE[1];
		cfg[0].DeviceInfo.DeviceType = GDS_DEVICE_TYPE_GNAUTILUS;

		strcpy( cfg[0].DeviceInfo.Name, device[0]);
		cfg[0].Configuration = config;

		//-------------------------------------------------------------------------------------
		// Apply the configuration on the server.
		//-------------------------------------------------------------------------------------
		ret = GDS_SetConfiguration( connectionHandle, cfg, 1 );
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Set the data ready callback threshold.
		//-------------------------------------------------------------------------------------
		size_t data_ready_threshold_number_of_scans = std::max<size_t>(config->SamplingRate * DATA_READY_THRESHOLD_MS/1000, config->NumberOfScans );
		ret = GDS_SetDataReadyCallback(connectionHandle, on_data_ready_event, data_ready_threshold_number_of_scans, (void*) &usrData);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Free the resources used by the config.
		//-------------------------------------------------------------------------------------
		delete config;
		delete [] cfg;


	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		Status->Message = ex->Message;
	}

}


void GtecInterface::CalibrateElectrodes(void) 
{

	Calibration_Values = gcnew array <String ^> (GDS_GNAUTILUS_CHANNELS_MAX); //Re-Initialize

	Status->Success = true;
	Status->Message = "";
	try
	{

		//-------------------------------------------------------------------------------------
		// Calibrate Electrode Offset and Scaling Factors
		//-------------------------------------------------------------------------------------

		GDS_GNAUTILUS_SCALING* scaling = new GDS_GNAUTILUS_SCALING;

		GDS_RESULT ret = GDS_GNAUTILUS_ResetScaling(connectionHandle, device);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		ret = GDS_GNAUTILUS_Calibrate(connectionHandle, device, scaling);

		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }
		else 
		{
			ret = GDS_GNAUTILUS_SetScaling(connectionHandle, device, scaling);
			if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
				throw gcnew Exception(gcnew String(ret.ErrorMessage)); }
			else 
			{
				for ( int i = 0; i < NumChannels; i++ )
				{
					Calibration_Values[i] = String::Format("{0}:\tOffset = {1}\tScaling Factor = {2}\n", Electrode_Names[i], scaling->Offset[i].ToString("00.00"), scaling->Factor[i].ToString("0.000")); 
				}
			}
		}

		delete scaling;
	
	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		Status->Message = ex->Message + "\n";
	}
	
}

void GtecInterface::CalculateImpedances(void) 
{

	Status->Success = true;
	Status->Message = "";

	try
	{
		
		//ResultStruct^ Res = ConnectToDevice();
		//if (Res->Success != true) {
		//throw gcnew Exception(Res->Message); }

		//Initialise Impedances to 0
		Impedance_Values = gcnew array <String ^> (GDS_GNAUTILUS_CHANNELS_MAX); //Initialize
		Impedances = new double[1][GDS_GNAUTILUS_CHANNELS_MAX]; //2D Array;
			
		GDS_RESULT ret = GDS_GNAUTILUS_GetImpedance(connectionHandle, device, Impedances);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }
		else
		{
			for ( int i = 0; i < NumChannels; i++ )
			{
				Impedance_Values[i] = String::Format("{0}:\t{1} kOhms", Electrode_Names[i], (Impedances[0][i]/1000).ToString("00.0"));
			}

		}
		

	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		Status->Message = ex->Message;
		
	}

	
}

void GtecInterface::RunScan(void) 
{

	Status->Success = true;
	Status->Message = "";
	GDS_RESULT ret;
	
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	Thread^ current = Thread::CurrentThread;
	current->Priority = ThreadPriority::Highest;

	StimSequence^ Stimulus = gcnew StimSequence();

	try
	{

		//-------------------------------------------------------------------------------------
		// Create Handles
		//-------------------------------------------------------------------------------------
			glb_event_handle = NULL;
			glb_event_handle = CreateEvent(NULL, false, false, NULL);
			if ( glb_event_handle == NULL ){
				throw gcnew Exception(GetLastError().ToString());}

			glb_mutex_handle = NULL;
			glb_mutex_handle = CreateMutex(NULL, FALSE, NULL);
			if ( glb_mutex_handle == NULL ) {
				throw gcnew Exception(GetLastError().ToString());}
		
		//-------------------------------------------------------------------------------------
		// Initialise buffer parameters
		//-------------------------------------------------------------------------------------

		size_t buffer_size_per_scan = 0;
		size_t scan_count = 1;
		size_t channels_per_device_count = 0;

		//-------------------------------------------------------------------------------------
		// Retrieve the buffer size for a single scan.
		//-------------------------------------------------------------------------------------
		ret = GDS_GetDataInfo( connectionHandle, &scan_count, NULL, &channels_per_device_count, &buffer_size_per_scan );
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		const static unsigned int data_rate = DATA_RATE;
		scan_count = (size_t) ( (data_rate * 1024)/(double)(buffer_size_per_scan * sizeof(float)) );
		size_t buffer_size_in_samples = scan_count * buffer_size_per_scan;

		uint64_t total_acquired_scans = 0;
		uint64_t total_scans_to_acquire = this->RunData->Duration * SAMPLE_RATE;

		DWORD dwWaitResult, dwOwnMutex;

		std::string DATA_FILE;
		DATA_FILE = msclr::interop::marshal_as<std::string>(this->RunData->FileName + this->RunData->Extension );  //saves to hard-disk location first

		//-------------------------------------------------------------------------------------
		// Command the device to acquire data.
		//-------------------------------------------------------------------------------------
		ret = GDS_StartAcquisition( connectionHandle );
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Stream the measurement data via the network.
		//-------------------------------------------------------------------------------------

		ret = GDS_StartStreaming( connectionHandle );
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		// Collect and Save EEG Data

		float* data_buffer = new float[buffer_size_in_samples];
		std::ofstream file( DATA_FILE, std::ios_base::binary );

		//Begin Stimulus Sequence
		while ( total_acquired_scans < total_scans_to_acquire )
		{
			// wait until the server signals that the specified amount of data is available.
			dwWaitResult = WaitForSingleObject(glb_event_handle, SYSTEM_EVENT_TIMEOUT);
			if (dwWaitResult != WAIT_OBJECT_0)
				throw gcnew Exception("The data ready event hasn't been triggered within a reasonable time.");

			// lock the mutex in order to enable the triggered data ready event method to ignore further incoming events.
			dwOwnMutex = WaitForSingleObject(glb_mutex_handle, SYSTEM_EVENT_TIMEOUT);
			if (dwOwnMutex != WAIT_OBJECT_0)
				throw gcnew Exception("Couldn't acquire the lock for the mutex which accompanies the data ready event.");

			size_t scans_available = scan_count;
			ret = GDS_GetData( connectionHandle, &scans_available, data_buffer, buffer_size_in_samples );
			if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
				throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

			if ( scans_available > 0 )
			{
				total_acquired_scans += scans_available;
				file.write( (const char*) data_buffer, scans_available * buffer_size_per_scan * sizeof(float) );

			}
			ReleaseMutex(glb_mutex_handle);
		}

	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		String^ m = ex->Message + "\n";
		Status->Message = m;
		ReleaseMutex(glb_mutex_handle);
	}

	try
	{
		//-------------------------------------------------------------------------------------
		// Free the buffer and close the file.
		//-------------------------------------------------------------------------------------
		__if_exists (data_buffer) {
			delete[] data_buffer;
			data_buffer = NULL; }
		__if_exists (file) {
			file.close(); }

		//-------------------------------------------------------------------------------------
		// Stop streaming data via the network.
		//-------------------------------------------------------------------------------------
		ret = GDS_StopStreaming(connectionHandle);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Command the device to stop acquiring data.
		//-------------------------------------------------------------------------------------
		ret = GDS_StopAcquisition(connectionHandle);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		String^ m = Status->Message + ex->Message + "\n";
		Status->Message = m;
		ReleaseMutex(glb_mutex_handle);

	}

}

void GtecInterface::Disconnect(void) 
{
	Status->Success = true;
	Status->Message = "";

	try
	{
		//-------------------------------------------------------------------------------------
		// If the global handles exist then close them.
		//-------------------------------------------------------------------------------------
		if (glb_event_handle)
		{
			CloseHandle(glb_event_handle);
			glb_event_handle = NULL;
		}

		if (glb_mutex_handle)
		{
			CloseHandle(glb_mutex_handle);
			glb_mutex_handle = NULL;
		}

		//-------------------------------------------------------------------------------------
		// Disconnect from the server.
		//-------------------------------------------------------------------------------------
		GDS_RESULT ret = GDS_Disconnect(&connectionHandle);
		if (ret.ErrorCode != GDS_ERROR_SUCCESS ) {
			throw gcnew Exception(gcnew String(ret.ErrorMessage)); }

		//-------------------------------------------------------------------------------------
		// Uninitialize the library.
		//-------------------------------------------------------------------------------------
		GDS_Uninitialize();

	}
	catch (Exception^ ex)
	{
		Status->Success = false;
		Status->Message = ex->Message;
	}


}

void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData)
{
	
	SetEvent(glb_event_handle);
	
	/* //-------------------------------------------------------------------------------------
	// The method is triggered periodically by the server as long as the specified
	// amount of data is available.
	//-------------------------------------------------------------------------------------

	DWORD dwOwnMutex = WaitForSingleObject(glb_mutex_handle, 0);
	// if daq is busy, then ignore further incoming events.
	if (dwOwnMutex != WAIT_OBJECT_0)
	{
		if (dwOwnMutex == WAIT_ABANDONED)
		{
			ReleaseMutex(glb_mutex_handle);
			glb_mutex_handle = NULL;
		}
		return;
	}

	if ( !SetEvent(glb_event_handle) ){

		//NeuroCatch::MainForm::terminal->AppendText(System::String::Format("ERROR: SetEvent(event) failed with error code {0}\n\n", GetLastError()));
		//NeuroCatch::MainForm::terminal->Refresh();
	}
	ReleaseMutex(glb_mutex_handle); */
}

void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData)
{
	//-------------------------------------------------------------------------------------
	// This method will be involved if the data acquisition struggles.
	//-------------------------------------------------------------------------------------
	//std::clog << "------------------------------------" << std::endl;
	//std::clog << "Handle			= " << connectionHandle << std::endl;
	//std::clog << "Device			= " << usrData << std::endl;
	//std::clog << "Where			= onDataAcquisitionError" << std::endl;
	//std::clog << "What			= " << result.ErrorMessage << std::endl;
	//std::clog << "ErrorCode		= " << result.ErrorCode << std::endl;
	//std::clog << std::endl;
}

void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData)
{
	//-------------------------------------------------------------------------------------
	// This method will be involved if the server dies.
	//-------------------------------------------------------------------------------------

	//std::clog << "------------------------------------" << std::endl;
	//std::clog << "Handle = " << connectionHandle << std::endl;
	//std::clog << "What   = onServerDiedEvent" << std::endl; 
}