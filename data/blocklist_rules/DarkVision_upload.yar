rule DarkVision_upload
{

    meta:
        description = "DarkVision_upload"
        author = "01Xyris"
        date = "2024-11-25"
      yarahub_reference_md5 = "0f2800e7a761d58fc3d25abfce6a7e8e"
      yarahub_uuid = "7e558990-e85d-4e12-9a07-1a70b0413f50"
      yarahub_license = "CC0 1.0"
      yarahub_rule_matching_tlp = "TLP:WHITE"
      yarahub_rule_sharing_tlp = "TLP:WHITE"
        
    strings:
        $upload_php_utf16le = { 75 00 70 00 6C 00 6F 00 61 00 64 00 2E 00 70 00 68 00 70 00 }

    condition:
        $upload_php_utf16le
}
