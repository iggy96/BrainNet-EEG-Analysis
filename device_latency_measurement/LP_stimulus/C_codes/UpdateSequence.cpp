//NC with BRO


#include "UpdateSequence.h"

UpdateSequence::UpdateSequence()
{
	Sequence = gcnew SequenceStruct();
}

SequenceStruct^ UpdateSequence::Run(ContextStruct^ ContextIn, int Type, int Rows, int NumTones)
{
	//type is an integer that specifies variations of the neurocatch sequence
	// type = 0    -> NeuroCatch
	// type = 1    -> NeuroCatch youth version	
	
	srand((unsigned)time(0)); 
	int Choose	 = (int)(rand()%2)+5;  //Random integer between 2 and 7

	String^ SeqNum = Choose.ToString();  
	String^ StimFile;
	String^ Extension;

	switch (Type)
	{
	case 0:
		StimFile = "4tones20NC2.txt";
		break;
	case 1:
		StimFile = "4tones20NC3.txt";
		break;
	case 2:
		StimFile = "4tones20NC4.txt";
		break;
	}

	//StimFile = "NC" + SeqNum + Extension;
	//StimFile = "4tones6NC" + SeqNum + Extension;

	//Change file Name

	Sequence->Type = StimFile;
	Sequence->Rows = Rows;
	Sequence->NumTones = NumTones;
	
	String^ linein;
	StreamReader^ NC = gcnew StreamReader("Stim\\" + StimFile);

	Sequence->Data = gcnew array <String^, 2>(Rows, (NumTones*2)+4);

	String^ delimStr = ",";
	array<Char>^ delimiter = delimStr->ToCharArray( );
	array<String^>^ tokens;

	for (int i = 0; i < Rows; i++)
	{
		linein = NC->ReadLine();
		tokens = linein->Split( delimiter );

		if (String::Compare(tokens[(NumTones*2)], "day") == 0)
			tokens[(NumTones*2)] = ContextIn->Day->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "month") == 0)
			tokens[(NumTones*2)] = ContextIn->Month->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "year") == 0)
			tokens[(NumTones*2)] = ContextIn->Year->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "season") == 0)
			tokens[(NumTones*2)] = ContextIn->Season->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "time") == 0)
			tokens[(NumTones*2)] = ContextIn->Time->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "location") == 0)
			tokens[(NumTones*2)] = ContextIn->Location->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "city") == 0)
			tokens[(NumTones*2)] = ContextIn->City->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "country") == 0)
			tokens[(NumTones*2)] = ContextIn->Country->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "continent") == 0)
			tokens[(NumTones*2)] = ContextIn->Continent->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "province") == 0)
			tokens[(NumTones*2)] = ContextIn->State->ToLower();
		else if (String::Compare(tokens[(NumTones*2)], "state") == 0)
			tokens[(NumTones*2)] = ContextIn->State->ToLower();

		for (int j=0; j < (tokens->Length); j++)
		{
			Sequence->Data[i,j] = tokens[j];
		}

	}


	return Sequence;


} // end method 