#pragma once
#include "PlaySequence.h"
#include <stdio.h>
#include <Windows.h>

StimSequence::StimSequence()
{

}

//controls Thread that plays the stimulus sequence
void StimSequence::Play()
{
	//// obtain reference to currently executing thread
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	Thread^ current = Thread::CurrentThread;
	current->Priority = ThreadPriority::Highest;
	timeBeginPeriod(10);

	String^ FileName;
	LPCTSTR Play;
	marshal_context context;

	int Delay1, Delay2, Delay3, Delay4;

	switch (Sequence->NumTones)
	{
	case 3:
	case 4:
		Delay1 = 500;
		break;
	case 5:
		Delay1 = 400;
		break;
	default:
		Delay1 = 400;
	}
	
	Delay2 = Delay1 + 100;
	Delay3 = 1400;
	Delay4 = 1500;

	DWORD ret;
	FT_STATUS ftStatus = NULL;
	ftStatus = FT_Open(0, &handle);
	//Todo - Error checking
	current->Sleep(50);
	ftStatus = FT_SetBitMode(handle, 0xFF,1);
	current->Sleep(50);

	char Trigger[1];

	//Set the Port to 0
	Trigger[0] = (char)0;
    FT_Write(handle, Trigger, 1, &ret);

	int Code;

	//Play Empty File
	FileName = "Audio\\Empty.wav";
	Play = context.marshal_as<const TCHAR*>(FileName);
	PlaySound(Play, NULL, SND_ASYNC);

	//5 Sec delay before tones start - time for device to set up
	current->Sleep(5000); 

	for (int i = 0; i < Sequence->Rows; i++)
	{
		for (int j = 0; j < (Sequence->NumTones+2); j++)
		{
			FileName = "Audio\\" + Sequence->Data[i,(j*2)] + ".wav";

			Code = int::Parse(Sequence->Data[i, (j*2)+1]);
			Trigger[0] = (char)Code;
			FT_Write(handle, Trigger, 1, &ret);

			Play = context.marshal_as<const TCHAR*>(FileName);
			PlaySound(Play, NULL, SND_ASYNC);

			current->Sleep(100);

			//Set the Port to 0
			Trigger[0] = (char)0;
			FT_Write(handle, Trigger, 1, &ret);

			if (j < Sequence->NumTones - 1) 
				current->Sleep(Delay1-100);
			else if (j == Sequence->NumTones - 1) 
				current->Sleep(Delay2-100);
			else if (j == Sequence->NumTones)
				current->Sleep(Delay3-100);
			else 
				current->Sleep(Delay4-100);

		}

	}
	
	timeEndPeriod(10);
	FT_Close(handle);

} // end method Play

void StimSequence::Connection_Test()
{
	DWORD ret;
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	Thread^ current = Thread::CurrentThread;
	current->Priority = ThreadPriority::Highest;
	
	FT_STATUS ftStatus = NULL;
	ftStatus = FT_Open(0, &handle);
	current->Sleep(50);
	ftStatus = FT_SetBitMode(handle, 0xFF,1);
	current->Sleep(50);

	char Trigger[1];
	Trigger[0] = (char)0;	
    FT_Write(handle, Trigger, 1, &ret);
	current->Sleep(3500);  	//3.5 Sec delay before 3sec test starts - time for device to set up if necessary

	for (int j = 0; j < 15; j++)
	{
		Trigger[0] = (char)j;	
		FT_Write(handle, Trigger, 1, &ret);

		current->Sleep(100);
		Trigger[0] = (char)0;
		FT_Write(handle, Trigger, 1, &ret);
		current->Sleep(100);
	}

}


void StimSequence::Trigger_Latency_Test()
{
	DWORD ret;
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
	Thread^ current = Thread::CurrentThread;
	current->Priority = ThreadPriority::Highest;
	
	FT_STATUS ftStatus = NULL;
	ftStatus = FT_Open(0, &handle);
	current->Sleep(50);
	ftStatus = FT_SetBitMode(handle, 0xFF,1);
	current->Sleep(50);

	String^ FileName;
	LPCTSTR Play;
	marshal_context context;
	FileName = "Audio\\Empty.wav";
	Play = context.marshal_as<const TCHAR*>(FileName);
	PlaySound(Play, NULL, SND_ASYNC);

	char Trigger[1];
	Trigger[0] = (char)0;	
    FT_Write(handle, Trigger, 1, &ret);
	current->Sleep(5000);  	//3.5 Sec delay before 3sec test starts - time for device to set up if necessary

	FileName = "Audio\\test_30Hz_250ms.wav";
	for (int j = 0; j < 100; j++)
	{
		Trigger[0] = (char)1;	
		Play = context.marshal_as<const TCHAR*>(FileName);	
	
		PlaySound(Play, NULL, SND_ASYNC);
		FT_Write(handle, Trigger, 1, &ret);
	

		current->Sleep(100);
		Trigger[0] = (char)0;
		FT_Write(handle, Trigger, 1, &ret);
		current->Sleep(400);
	}


}


void StimSequence::Play_Sample()
{
		Thread^ current = Thread::CurrentThread;
		String^ FileName;
		LPCTSTR Play;
		marshal_context context;
		FileName = "Audio\\Empty.wav";
		Play = context.marshal_as<const TCHAR*>(FileName);
		PlaySound(Play, NULL, SND_SYNC);
		current->Sleep(1000);  	//3.5 Sec delay before 3sec test starts - time for device to set up if necessary
		FileName = "Audio\\Sample.wav";
		Play = context.marshal_as<const TCHAR*>(FileName);	
		PlaySound(Play, NULL, SND_SYNC);

}
