import "pe"
import "math"

rule Sus_Obf_Enc_Spoof_Hide_PE {

    meta:
        author = "XiAnzheng"
        source_url = "https://github.com/XiAnzheng-ID/Yara-Rules"
        description = "Check for Overlay, Obfuscating, Encrypting, Spoofing, Hiding, or Entropy Technique(can create FP)"
        date = "2024-11-18"
        updated = "2025-02-08"
        yarahub_license = "CC0 1.0"
        yarahub_uuid = "fa466824-f124-45bc-8398-eaecef7271f9"
        yarahub_rule_matching_tlp = "TLP:WHITE"
        yarahub_rule_sharing_tlp = "TLP:WHITE"
        yarahub_reference_md5 = "ffea1266b09abbf0ceb59119746d8630"

    condition:
        // Missing, Suspicious, Spoofed Import/Export tables combination 
        (pe.number_of_imports == 0)
        or (pe.number_of_imports == 0 and pe.entry_point_raw == 0)
        or (pe.size_of_optional_header < 0xE0 or pe.size_of_optional_header > 0xF0)
        or (pe.number_of_exports != 0 and pe.number_of_imports == 0)

        // Suspicious or Spoofed Section Headers Number
        or (pe.number_of_sections == 0 or pe.number_of_sections < 0 or pe.number_of_sections > 11)

        // Contain Overlay File (Can create FP)
        or (pe.overlay.size > 0)

        // Invalid PE Header
        or (pe.size_of_headers < 0x200 or pe.size_of_headers > 0x400)

        // High Entropy Section (Could Be Compressed or Packed, Can Create FP)
        or (math.entropy(0, filesize) > 7.25)
        
        or (for any ent_sec in (0..pe.number_of_sections - 1): (
                math.entropy(pe.sections[ent_sec].raw_data_offset, pe.sections[ent_sec].raw_data_offset + pe.sections[ent_sec].raw_data_size) > 7.25
            )
        )
}