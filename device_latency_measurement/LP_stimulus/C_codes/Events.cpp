#include "stdafx.h"
#include "MainForm.h"
#include "GDSClientAPI.h"
#include "GDSClientAPI_gNautilus.h"

//-------------------------------------------------------------------------------------
// The method is triggered periodically by the server as long as the specified
// amount of data is available.
//-------------------------------------------------------------------------------------
void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData)
{
	NeuroCatch::MainForm::terminal->Text = "On Data Ready Event Is Working...";
	NeuroCatch::MainForm::terminal->Refresh();
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
	ReleaseMutex(glb_mutex_handle);
}

		
//-------------------------------------------------------------------------------------
// This method will be involved if the data acquisition struggles.
//-------------------------------------------------------------------------------------
void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData)
{
	//std::clog << "------------------------------------" << std::endl;
	//std::clog << "Handle			= " << connectionHandle << std::endl;
	//std::clog << "Device			= " << usrData << std::endl;
	//std::clog << "Where			= onDataAcquisitionError" << std::endl;
	//std::clog << "What			= " << result.ErrorMessage << std::endl;
	//std::clog << "ErrorCode		= " << result.ErrorCode << std::endl;
	//std::clog << std::endl;
}

//-------------------------------------------------------------------------------------
// This method will be involved if the server dies.
//-------------------------------------------------------------------------------------
void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData)
{
	//std::clog << "------------------------------------" << std::endl;
	//std::clog << "Handle = " << connectionHandle << std::endl;
	//std::clog << "What   = onServerDiedEvent" << std::endl; 
}
